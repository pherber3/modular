// ===----------------------------------------------------------------------=== //
// TDT Persistent Megakernel — CUDA C++ Diagnostic Benchmark
//
// Direct 1:1 translation of the Mojo megakernel to CUDA C++.
// Used to determine if the 170x slowdown in Mojo is a compiler issue.
//
// Compile: nvcc -O3 -arch=sm_89 -std=c++17 tdt_megakernel_cuda.cu -o tdt_megakernel_cuda
// Run:     python3 dump_weights.py && ./tdt_megakernel_cuda
// ===----------------------------------------------------------------------=== //

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <algorithm>
#include <chrono>
#include <vector>

// TDT decoder dimensions (parakeet-tdt-0.6b-v3)
#define PRED_HIDDEN 640
#define GATES_DIM 2560
#define JOINT_HIDDEN 640
#define NUM_TOKENS 8193
#define BLANK_ID 8192
#define NUM_DURATIONS 5
#define OUTPUT_SIZE 8198
#define BLOCK_SIZE 256
#define MAX_ENCODER_FRAMES 400
#define MAX_OUTPUT_TOKENS 4096
#define MAX_SYMBOLS_PER_STEP 10
#define MAX_OUTER_ITERS 10000

#define CEILDIV(a, b) (((a) + (b) - 1) / (b))

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)


// ====================================================================
// Single decode step — called from both SOS and main loop
// ====================================================================
__device__ void run_decode_step(
    int tid, int tok, int t_val,
    // Shared memory
    float *x_buf, float *gates_buf,
    float *h0_buf, float *c0_buf,
    float *h1_buf, float *c1_buf,
    float *joint_buf, int *ctrl,
    // Global memory
    const float *enc_projected, const float *embedding,
    const float *l0_ih_w, const float *l0_ih_b,
    const float *l0_hh_w, const float *l0_hh_b,
    const float *l1_ih_w, const float *l1_ih_b,
    const float *l1_hh_w, const float *l1_hh_b,
    const float *pred_w, const float *pred_b,
    const float *out_w, const float *out_b
) {
    // ---- Embedding lookup ----
    for (int i = 0; i < CEILDIV(PRED_HIDDEN, BLOCK_SIZE); i++) {
        int idx = i * BLOCK_SIZE + tid;
        if (idx < PRED_HIDDEN)
            x_buf[idx] = embedding[tok * PRED_HIDDEN + idx];
    }
    __syncthreads();

    // ---- LSTM Layer 0: ih GEMV ----
    // W layout: [j][k] = W[j * PRED_HIDDEN + k], j = output, k = input
    for (int j_off = 0; j_off < CEILDIV(GATES_DIM, BLOCK_SIZE); j_off++) {
        int j = j_off * BLOCK_SIZE + tid;
        if (j < GATES_DIM) {
            float acc = l0_ih_b[j];
            for (int k = 0; k < PRED_HIDDEN; k++)
                acc += x_buf[k] * l0_ih_w[j * PRED_HIDDEN + k];
            gates_buf[j] = acc;
        }
    }
    __syncthreads();

    // ---- LSTM Layer 0: hh GEMV-add ----
    for (int j_off = 0; j_off < CEILDIV(GATES_DIM, BLOCK_SIZE); j_off++) {
        int j = j_off * BLOCK_SIZE + tid;
        if (j < GATES_DIM) {
            float acc = gates_buf[j] + l0_hh_b[j];
            for (int k = 0; k < PRED_HIDDEN; k++)
                acc += h0_buf[k] * l0_hh_w[j * PRED_HIDDEN + k];
            gates_buf[j] = acc;
        }
    }
    __syncthreads();

    // ---- LSTM Layer 0: gate activations -> h0_buf, c0_buf ----
    for (int i_off = 0; i_off < CEILDIV(PRED_HIDDEN, BLOCK_SIZE); i_off++) {
        int idx = i_off * BLOCK_SIZE + tid;
        if (idx < PRED_HIDDEN) {
            float gi = gates_buf[idx];
            float gf = gates_buf[PRED_HIDDEN + idx];
            float gg = gates_buf[2 * PRED_HIDDEN + idx];
            float go = gates_buf[3 * PRED_HIDDEN + idx];

            float sig_i = 1.0f / (1.0f + expf(-gi));
            float sig_f = 1.0f / (1.0f + expf(-gf));
            float tanh_g = tanhf(gg);
            float sig_o = 1.0f / (1.0f + expf(-go));

            float c_new = sig_f * c0_buf[idx] + sig_i * tanh_g;
            float h_new = sig_o * tanhf(c_new);

            h0_buf[idx] = h_new;
            c0_buf[idx] = c_new;
        }
    }
    __syncthreads();

    // ---- Prepare L1 input ----
    for (int i = 0; i < CEILDIV(PRED_HIDDEN, BLOCK_SIZE); i++) {
        int idx = i * BLOCK_SIZE + tid;
        if (idx < PRED_HIDDEN)
            x_buf[idx] = h0_buf[idx];
    }
    __syncthreads();

    // ---- LSTM Layer 1: ih GEMV ----
    for (int j_off = 0; j_off < CEILDIV(GATES_DIM, BLOCK_SIZE); j_off++) {
        int j = j_off * BLOCK_SIZE + tid;
        if (j < GATES_DIM) {
            float acc = l1_ih_b[j];
            for (int k = 0; k < PRED_HIDDEN; k++)
                acc += x_buf[k] * l1_ih_w[j * PRED_HIDDEN + k];
            gates_buf[j] = acc;
        }
    }
    __syncthreads();

    // ---- LSTM Layer 1: hh GEMV-add ----
    for (int j_off = 0; j_off < CEILDIV(GATES_DIM, BLOCK_SIZE); j_off++) {
        int j = j_off * BLOCK_SIZE + tid;
        if (j < GATES_DIM) {
            float acc = gates_buf[j] + l1_hh_b[j];
            for (int k = 0; k < PRED_HIDDEN; k++)
                acc += h1_buf[k] * l1_hh_w[j * PRED_HIDDEN + k];
            gates_buf[j] = acc;
        }
    }
    __syncthreads();

    // ---- LSTM Layer 1: gate activations -> h1_buf, c1_buf ----
    for (int i_off = 0; i_off < CEILDIV(PRED_HIDDEN, BLOCK_SIZE); i_off++) {
        int idx = i_off * BLOCK_SIZE + tid;
        if (idx < PRED_HIDDEN) {
            float gi = gates_buf[idx];
            float gf = gates_buf[PRED_HIDDEN + idx];
            float gg = gates_buf[2 * PRED_HIDDEN + idx];
            float go = gates_buf[3 * PRED_HIDDEN + idx];

            float sig_i = 1.0f / (1.0f + expf(-gi));
            float sig_f = 1.0f / (1.0f + expf(-gf));
            float tanh_g = tanhf(gg);
            float sig_o = 1.0f / (1.0f + expf(-go));

            float c_new = sig_f * c1_buf[idx] + sig_i * tanh_g;
            float h_new = sig_o * tanhf(c_new);

            h1_buf[idx] = h_new;
            c1_buf[idx] = c_new;
        }
    }
    __syncthreads();

    // ---- Joint: pred_proj ----
    for (int j_off = 0; j_off < CEILDIV(JOINT_HIDDEN, BLOCK_SIZE); j_off++) {
        int j = j_off * BLOCK_SIZE + tid;
        if (j < JOINT_HIDDEN) {
            float acc = pred_b[j];
            for (int k = 0; k < PRED_HIDDEN; k++)
                acc += h1_buf[k] * pred_w[j * PRED_HIDDEN + k];
            joint_buf[j] = acc;
        }
    }
    __syncthreads();

    // ---- Joint: relu(enc_t + pred_projected) ----
    for (int i = 0; i < CEILDIV(JOINT_HIDDEN, BLOCK_SIZE); i++) {
        int idx = i * BLOCK_SIZE + tid;
        if (idx < JOINT_HIDDEN) {
            float val = joint_buf[idx] + enc_projected[t_val * JOINT_HIDDEN + idx];
            joint_buf[idx] = fmaxf(val, 0.0f);
        }
    }
    __syncthreads();

    // ---- Copy joint_buf to x_buf for output_proj ----
    for (int i = 0; i < CEILDIV(JOINT_HIDDEN, BLOCK_SIZE); i++) {
        int idx = i * BLOCK_SIZE + tid;
        if (idx < JOINT_HIDDEN)
            x_buf[idx] = joint_buf[idx];
    }
    __syncthreads();

    // ---- Fused output_proj GEMV + argmax ----
    float local_token_max_val = -1e30f;
    int local_token_max_idx = 0;
    float local_dur_max_val = -1e30f;
    int local_dur_max_idx = 0;

    for (int j_off = 0; j_off < CEILDIV(OUTPUT_SIZE, BLOCK_SIZE); j_off++) {
        int j = j_off * BLOCK_SIZE + tid;
        if (j < OUTPUT_SIZE) {
            float acc = out_b[j];
            for (int k = 0; k < JOINT_HIDDEN; k++)
                acc += x_buf[k] * out_w[j * JOINT_HIDDEN + k];
            if (j < NUM_TOKENS) {
                if (acc > local_token_max_val) {
                    local_token_max_val = acc;
                    local_token_max_idx = j;
                }
            } else {
                if (acc > local_dur_max_val) {
                    local_dur_max_val = acc;
                    local_dur_max_idx = j - NUM_TOKENS;
                }
            }
        }
    }

    // Warp-level reduction for token argmax
    for (int offset = 0; offset < 5; offset++) {
        float shuf_val = __shfl_down_sync(0xffffffff, local_token_max_val, 1 << (4 - offset));
        int shuf_idx = __shfl_down_sync(0xffffffff, local_token_max_idx, 1 << (4 - offset));
        if (shuf_val > local_token_max_val) {
            local_token_max_val = shuf_val;
            local_token_max_idx = shuf_idx;
        }
    }

    // Warp-level reduction for duration argmax
    for (int offset = 0; offset < 5; offset++) {
        float shuf_val = __shfl_down_sync(0xffffffff, local_dur_max_val, 1 << (4 - offset));
        int shuf_idx = __shfl_down_sync(0xffffffff, local_dur_max_idx, 1 << (4 - offset));
        if (shuf_val > local_dur_max_val) {
            local_dur_max_val = shuf_val;
            local_dur_max_idx = shuf_idx;
        }
    }

    // Lane 0 of each warp writes to gates_buf
    if (tid % 32 == 0) {
        int warp_slot = tid / 32;
        gates_buf[warp_slot] = local_token_max_val;
        // Store index as float
        gates_buf[8 + warp_slot] = __int_as_float(local_token_max_idx);
        gates_buf[16 + warp_slot] = local_dur_max_val;
        gates_buf[24 + warp_slot] = __int_as_float(local_dur_max_idx);
    }
    __syncthreads();

    // Thread 0: final cross-warp argmax
    if (tid == 0) {
        float best_val = -1e30f;
        int best_idx = 0;
        for (int w = 0; w < CEILDIV(BLOCK_SIZE, 32); w++) {
            float wval = gates_buf[w];
            if (wval > best_val) {
                best_val = wval;
                best_idx = __float_as_int(gates_buf[8 + w]);
            }
        }
        float dur_best_val = -1e30f;
        int dur_best_idx = 0;
        for (int w = 0; w < CEILDIV(BLOCK_SIZE, 32); w++) {
            float dval = gates_buf[16 + w];
            if (dval > dur_best_val) {
                dur_best_val = dval;
                dur_best_idx = __float_as_int(gates_buf[24 + w]);
            }
        }
        ctrl[0] = best_idx;      // best_token
        ctrl[1] = dur_best_idx;   // best_dur_idx
    }
    __syncthreads();
}


// ====================================================================
// Main megakernel
// ====================================================================
extern "C" __global__ void tdt_megakernel(
    int *output_tokens, int *output_count,
    const float *enc_projected, const float *embedding, const int *durations,
    const float *l0_ih_w, const float *l0_ih_b,
    const float *l0_hh_w, const float *l0_hh_b,
    const float *l1_ih_w, const float *l1_ih_b,
    const float *l1_hh_w, const float *l1_hh_b,
    const float *pred_w, const float *pred_b,
    const float *out_w, const float *out_b
) {
    int tid = threadIdx.x;

    __shared__ float x_buf[PRED_HIDDEN];
    __shared__ float gates_buf[GATES_DIM];
    __shared__ float h0_buf[PRED_HIDDEN];
    __shared__ float c0_buf[PRED_HIDDEN];
    __shared__ float h1_buf[PRED_HIDDEN];
    __shared__ float c1_buf[PRED_HIDDEN];
    __shared__ float joint_buf[JOINT_HIDDEN];
    __shared__ int ctrl[8];
    __shared__ int total_steps;

    // ---- Initialize ----
    for (int i = 0; i < CEILDIV(PRED_HIDDEN, BLOCK_SIZE); i++) {
        int idx = i * BLOCK_SIZE + tid;
        if (idx < PRED_HIDDEN) {
            h0_buf[idx] = 0.0f;
            c0_buf[idx] = 0.0f;
            h1_buf[idx] = 0.0f;
            c1_buf[idx] = 0.0f;
        }
    }
    if (tid == 0) {
        ctrl[2] = 0;          // t
        ctrl[3] = 0;          // token_count
        ctrl[4] = BLANK_ID;   // current_token
        total_steps = 0;
    }
    __syncthreads();

    // ---- SOS step ----
    run_decode_step(tid, BLANK_ID, 0,
        x_buf, gates_buf, h0_buf, c0_buf, h1_buf, c1_buf, joint_buf, ctrl,
        enc_projected, embedding,
        l0_ih_w, l0_ih_b, l0_hh_w, l0_hh_b,
        l1_ih_w, l1_ih_b, l1_hh_w, l1_hh_b,
        pred_w, pred_b, out_w, out_b);

    // ---- Main decode loop ----
    bool outer_done = false;
    for (int outer_iter = 0; outer_iter < MAX_OUTER_ITERS && !outer_done; outer_iter++) {
        int t_val = ctrl[2];
        if (t_val >= MAX_ENCODER_FRAMES) {
            outer_done = true;
            continue;
        }

        if (tid == 0) {
            ctrl[5] = 0;  // break_inner
            ctrl[6] = 0;  // symbols_at_t
        }
        __syncthreads();

        bool inner_broke = false;
        for (int inner_iter = 0; inner_iter < MAX_SYMBOLS_PER_STEP && !inner_broke; inner_iter++) {
            run_decode_step(tid, ctrl[4], t_val,
                x_buf, gates_buf, h0_buf, c0_buf, h1_buf, c1_buf, joint_buf, ctrl,
                enc_projected, embedding,
                l0_ih_w, l0_ih_b, l0_hh_w, l0_hh_b,
                l1_ih_w, l1_ih_b, l1_hh_w, l1_hh_b,
                pred_w, pred_b, out_w, out_b);

            if (tid == 0) total_steps++;

            // Thread 0: branch logic
            if (tid == 0) {
                int token = ctrl[0];
                int dur_idx = ctrl[1];
                int duration = durations[dur_idx];
                int tc = ctrl[3];

                if (token == BLANK_ID) {
                    int advance = duration < 1 ? 1 : duration;
                    ctrl[2] = t_val + advance;
                    ctrl[5] = 1;
                } else {
                    if (tc < MAX_OUTPUT_TOKENS)
                        output_tokens[tc] = token;
                    ctrl[3] = tc + 1;
                    ctrl[4] = token;
                    ctrl[6] = ctrl[6] + 1;
                    if (duration > 0) {
                        ctrl[2] = t_val + duration;
                        ctrl[5] = 1;
                    } else {
                        ctrl[5] = 0;
                    }
                }
            }
            __syncthreads();

            if (ctrl[5] == 1)
                inner_broke = true;
        }

        // max_symbols_per_step reached without break
        if (!inner_broke && !outer_done) {
            if (tid == 0)
                ctrl[2] = ctrl[2] + 1;
            __syncthreads();
        }
    }

    if (tid == 0) {
        output_count[0] = ctrl[3];
        // Diagnostic: write total steps and final t to end of output buffer
        output_tokens[MAX_OUTPUT_TOKENS - 1] = total_steps;
        output_tokens[MAX_OUTPUT_TOKENS - 2] = ctrl[2];  // final t
    }
}


// ====================================================================
// Host code: load weights, launch kernel, benchmark
// ====================================================================

float* load_bin(const char* path, size_t num_elements) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    float* data = (float*)malloc(num_elements * sizeof(float));
    size_t read = fread(data, sizeof(float), num_elements, f);
    fclose(f);
    if (read != num_elements) {
        fprintf(stderr, "Read %zu elements from %s, expected %zu\n", read, path, num_elements);
        exit(1);
    }
    return data;
}

int* load_bin_int(const char* path, size_t num_elements) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int* data = (int*)malloc(num_elements * sizeof(int));
    size_t read = fread(data, sizeof(int), num_elements, f);
    fclose(f);
    if (read != num_elements) {
        fprintf(stderr, "Read %zu elements from %s, expected %zu\n", read, path, num_elements);
        exit(1);
    }
    return data;
}

float* to_gpu(float* host, size_t n) {
    float* dev;
    CHECK_CUDA(cudaMalloc(&dev, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dev, host, n * sizeof(float), cudaMemcpyHostToDevice));
    return dev;
}

int* to_gpu_int(int* host, size_t n) {
    int* dev;
    CHECK_CUDA(cudaMalloc(&dev, n * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(dev, host, n * sizeof(int), cudaMemcpyHostToDevice));
    return dev;
}

int main() {
    const char* dir = "weights";

    printf("Loading weights from %s/...\n", dir);
    char path[256];

    #define LOAD_F(name, n) \
        snprintf(path, sizeof(path), "%s/%s.bin", dir, #name); \
        float* h_##name = load_bin(path, n); \
        float* d_##name = to_gpu(h_##name, n);

    #define LOAD_I(name, n) \
        snprintf(path, sizeof(path), "%s/%s.bin", dir, #name); \
        int* h_##name = load_bin_int(path, n); \
        int* d_##name = to_gpu_int(h_##name, n);

    LOAD_F(enc_projected, MAX_ENCODER_FRAMES * JOINT_HIDDEN)
    LOAD_F(embedding, NUM_TOKENS * PRED_HIDDEN)
    LOAD_I(durations, NUM_DURATIONS)
    LOAD_F(l0_ih_w, GATES_DIM * PRED_HIDDEN)
    LOAD_F(l0_ih_b, GATES_DIM)
    LOAD_F(l0_hh_w, GATES_DIM * PRED_HIDDEN)
    LOAD_F(l0_hh_b, GATES_DIM)
    LOAD_F(l1_ih_w, GATES_DIM * PRED_HIDDEN)
    LOAD_F(l1_ih_b, GATES_DIM)
    LOAD_F(l1_hh_w, GATES_DIM * PRED_HIDDEN)
    LOAD_F(l1_hh_b, GATES_DIM)
    LOAD_F(pred_w, JOINT_HIDDEN * PRED_HIDDEN)
    LOAD_F(pred_b, JOINT_HIDDEN)
    LOAD_F(out_w, OUTPUT_SIZE * JOINT_HIDDEN)
    LOAD_F(out_b, OUTPUT_SIZE)

    // Output buffers
    int *d_output_tokens, *d_output_count;
    CHECK_CUDA(cudaMalloc(&d_output_tokens, MAX_OUTPUT_TOKENS * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output_count, sizeof(int)));

    printf("Weights loaded. Warming up...\n");

    // Warmup
    for (int i = 0; i < 5; i++) {
        CHECK_CUDA(cudaMemset(d_output_tokens, 0, MAX_OUTPUT_TOKENS * sizeof(int)));
        CHECK_CUDA(cudaMemset(d_output_count, 0, sizeof(int)));
        tdt_megakernel<<<1, BLOCK_SIZE>>>(
            d_output_tokens, d_output_count,
            d_enc_projected, d_embedding, d_durations,
            d_l0_ih_w, d_l0_ih_b, d_l0_hh_w, d_l0_hh_b,
            d_l1_ih_w, d_l1_ih_b, d_l1_hh_w, d_l1_hh_b,
            d_pred_w, d_pred_b, d_out_w, d_out_b);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Get token count from last warmup
    int token_count;
    CHECK_CUDA(cudaMemcpy(&token_count, d_output_count, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Tokens decoded: %d\n", token_count);

    // Print first 20 tokens
    if (token_count > 0) {
        int tokens[20];
        int n = token_count < 20 ? token_count : 20;
        CHECK_CUDA(cudaMemcpy(tokens, d_output_tokens, n * sizeof(int), cudaMemcpyDeviceToHost));
        printf("First %d tokens: ", n);
        for (int i = 0; i < n; i++) printf("%d ", tokens[i]);
        printf("\n");
    }

    // Diagnostic: read total steps and final t
    int diag[2];
    CHECK_CUDA(cudaMemcpy(&diag[0], d_output_tokens + MAX_OUTPUT_TOKENS - 1, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&diag[1], d_output_tokens + MAX_OUTPUT_TOKENS - 2, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Total decode steps: %d\n", diag[0]);
    printf("Final t value: %d (of %d encoder frames)\n", diag[1], MAX_ENCODER_FRAMES);

    // Benchmark
    const int ITERS = 100;
    std::vector<double> times(ITERS);

    for (int i = 0; i < ITERS; i++) {
        CHECK_CUDA(cudaMemset(d_output_tokens, 0, MAX_OUTPUT_TOKENS * sizeof(int)));
        CHECK_CUDA(cudaMemset(d_output_count, 0, sizeof(int)));

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        tdt_megakernel<<<1, BLOCK_SIZE>>>(
            d_output_tokens, d_output_count,
            d_enc_projected, d_embedding, d_durations,
            d_l0_ih_w, d_l0_ih_b, d_l0_hh_w, d_l0_hh_b,
            d_l1_ih_w, d_l1_ih_b, d_l1_hh_w, d_l1_hh_b,
            d_pred_w, d_pred_b, d_out_w, d_out_b);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        times[i] = ms;

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    std::sort(times.begin(), times.end());
    double median = times[ITERS / 2];
    double p95 = times[(int)(ITERS * 0.95)];
    double mean = 0;
    for (auto t : times) mean += t;
    mean /= ITERS;

    printf("\n============================================================\n");
    printf("TDT Persistent Megakernel — CUDA C++ Benchmark\n");
    printf("============================================================\n");
    printf("  Tokens decoded: %d\n", token_count);
    printf("  Total steps:    %d\n", diag[0]);
    printf("  Final t:        %d / %d\n", diag[1], MAX_ENCODER_FRAMES);
    printf("  Per-step:       %.2fus\n", (median * 1000.0) / diag[0]);
    printf("  Single launch:  median=%.2fms  mean=%.2fms  p95=%.2fms\n", median, mean, p95);
    printf("\n");
    printf("  Reference (Mojo megakernel, L4):        1621ms\n");
    printf("  Reference (Phase 2 step x 300, L4):     14.10ms\n");
    printf("  Reference (MAX graph, A100):             23ms\n");
    printf("  Reference (TRT, L40S):                   15ms\n");
    printf("  Target:                                  5-10ms\n");

    return 0;
}
