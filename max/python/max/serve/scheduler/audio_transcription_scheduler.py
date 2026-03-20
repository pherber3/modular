# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import logging
import queue
from dataclasses import dataclass

from max.interfaces import (
    AudioTranscriptionContext,
    AudioTranscriptionInputs,
    AudioTranscriptionOutput,
    MAXPullQueue,
    MAXPushQueue,
    RequestID,
    Scheduler,
    SchedulerResult,
)
from max.pipelines.lib import AudioTranscriptionPipelineType
from max.profiler import traced

from .base import SchedulerProgress

logger = logging.getLogger("max.serve")


@dataclass
class AudioTranscriptionSchedulerConfig:
    """Audio transcription scheduler configuration."""

    # The maximum number of requests that can be in the transcription batch.
    max_batch_size: int


class AudioTranscriptionScheduler(Scheduler):
    def __init__(
        self,
        scheduler_config: AudioTranscriptionSchedulerConfig,
        pipeline: AudioTranscriptionPipelineType,
        request_queue: MAXPullQueue[AudioTranscriptionContext],
        response_queue: MAXPushQueue[
            dict[RequestID, SchedulerResult[AudioTranscriptionOutput]]
        ],
        cancel_queue: MAXPullQueue[list[RequestID]],
    ) -> None:
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue

    @traced
    def _create_batch_to_execute(
        self,
    ) -> dict[RequestID, AudioTranscriptionContext]:
        max_batch_size = self.scheduler_config.max_batch_size

        batch: dict[RequestID, AudioTranscriptionContext] = {}

        # Synchronous draining
        while True:
            if len(batch) >= max_batch_size:
                break

            try:
                item = self.request_queue.get_nowait()
            except queue.Empty:
                break

            batch[item.request_id] = item

        return batch

    def run_iteration(self) -> SchedulerProgress:
        """Create batches and schedule them for transcription.

        Returns:
            SchedulerProgress: Indicates whether work was performed.
        """
        batch_to_execute = self._create_batch_to_execute()
        if len(batch_to_execute) == 0:
            return SchedulerProgress.NO_PROGRESS

        self._schedule_transcribe(batch_to_execute)
        return SchedulerProgress.MADE_PROGRESS

    @traced
    def _handle_terminated_responses(
        self,
        batch_executed: dict[RequestID, AudioTranscriptionContext],
        batch_response: dict[RequestID, AudioTranscriptionOutput],
    ) -> None:
        """Handle responses for terminated requests."""
        already_terminated = set()
        terminated = batch_executed.keys() - batch_response.keys()
        for req_id in terminated:
            if req_id in already_terminated:
                continue
            del batch_executed[req_id]
            already_terminated.add(req_id)

    @traced
    def _schedule_transcribe(
        self, batch_to_execute: dict[RequestID, AudioTranscriptionContext]
    ) -> None:
        # Execute the batch.
        batch_responses = self.pipeline.execute(
            AudioTranscriptionInputs(batches=[batch_to_execute])
        )
        # Remove terminated requests.
        self._handle_terminated_responses(batch_to_execute, batch_responses)
        # Send responses to the API process.
        self.response_queue.put_nowait(
            {
                request_id: SchedulerResult.create(response)
                for request_id, response in batch_responses.items()
            }
        )
