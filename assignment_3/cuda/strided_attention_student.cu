#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <type_traits> // Enable type selection utilities inside device code
#include <limits> // Provide numeric limits such as negative infinity for softmax masking
#include <cmath> // Supply math functions like sqrt required by the kernel

#include <c10/cuda/CUDAException.h>

// CUDA kernel for the forward pass of strided attention
//
// T: data type (e.g., float, double, half)
//
// Computes: output = Softmax((Q * K_strided^T) / sqrt(head_dim)) * V_strided
//
// Grid/Block Dimensions:
// - gridDim.x: batch_size * num_heads
// - gridDim.y: seq_len (for queries)
// - blockDim.x: Should be a power of 2, e.g., 128, 256. Represents threads per query token.
//
template <typename AccT>
__device__ inline AccT reciprocal_sqrt_int(const int value) { // Provide generic declaration for reciprocal sqrt helper
    return AccT(1) / sqrt(static_cast<AccT>(value)); // Fallback uses standard sqrt for arbitrary accumulator types
}

template <>
__device__ inline float reciprocal_sqrt_int<float>(const int value) { // Specialize reciprocal sqrt for float accumulators
    return rsqrtf(static_cast<float>(value)); // Use fast intrinsic reciprocal square root for float precision
}

template <>
__device__ inline double reciprocal_sqrt_int<double>(const int value) { // Specialize reciprocal sqrt for double accumulators
    return 1.0 / sqrt(static_cast<double>(value)); // Compute reciprocal via double-precision sqrt to preserve accuracy
}

// Inside the GPU hardware, threads are executed in groups of 32 threads, called warps.
// warpSize is a constant provided by CUDA (usually 32).
template <typename T>
// Warp-level sum reduction helper for arbitrary accumulator types
__device__ inline T warp_reduce_sum(T value) { 
    // __activemask returns a bitmask of which threads (lanes) in the warp are active.
    // If only threads 0-15 are active → mask looks like 0x0000FFFF (each bit represents a lane/thread).
    unsigned mask = __activemask(); 
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {

        // Moves (shuffles) a register value from a higher-numbered thread to a lower-numbered thread in the same warp.

        // For example, offset=1 means:
        // * Thread 0 gets thread 1's value,
        // * Thread 1 gets thread 2's value,
        // etc.

        // Used for reductions (sum, max, etc.) across warp threads without shared memory.
        // Works only inside a warp → super fast.
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

template <typename T>
__global__ void strided_attention_forward_kernel(
    const T* __restrict__ q_ptr,
    const T* __restrict__ k_ptr,
    const T* __restrict__ v_ptr,
    T* __restrict__ output_ptr,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int stride) {
    
    
    // Note:
    // 1. Each SM can execute many blocks concurrently
    // 2. Grid → Blocks → Threads
    //     a. You launch a grid of blocks.
    //     b. Each block runs on one SM (One SM runs one or more blocks).
    //     c. Each block has many threads (typically 128-1024).
    //     d. Threads inside a block can cooperate via shared memory and synchronization
    // 3. A block = one (batch, head, query) tile in my kernel.
    //     Inside that block:
    //     a. Many threads share the same (batch, head, query) but work on different parts of the attention computation.
    //     b. For example:
    //         i. Each thread might process a subset of the key positions.
    //         ii. Or each thread might process a subset of the head_dim (feature dimension).
    
    
    // Record the index of the current thread inside the block
    const int thread_idx = threadIdx.x; 

    // Compute the current index of (batch, head, query) (logical coordinates)
    // Cache the flattened (batch, head) identifier for this block (batch*head)
    const int block_head = blockIdx.x; 
    // Cache the query position handled by this block (seq_len)
    const int block_query = blockIdx.y; 
    // Recover the batch index by unflattening block index
    const int batch_idx = block_head / num_heads; 
    // Recover the attention head index associated with this block
    const int head_idx = block_head % num_heads; 
    // Alias query token index for clarity
    const int query_idx = block_query; 
    
    // Define the maximum block size used for shared memory allocations 
    // (match the number of threads per block)
    constexpr int max_block = 256; 

    // Select accumulator precision based on template type
    // If T is double, use double; otherwise, use float
    using acc_t = typename std::conditional<std::is_same<T, double>::value, double, float>::type; 
    
    // Allocate Shared Memory
    // Shared buffer that stores the active query vector for reuse
    // (on-chip memory visible to all threads in a block)
    __shared__ acc_t shared_query[max_block]; 
    // Shared buffer reused for reductions and temporary values
    __shared__ acc_t shared_buffer[max_block];
    // Shared array holding attention scores or probabilities per stride location
    __shared__ acc_t shared_scores[max_block]; 
    // Shared scalar storing the softmax denominator
    __shared__ acc_t shared_denom; 
    
    // Compute flattened offset and then pointer into global memory of each (batch, head, query)
    // tensor (B,H,N,D) is stored in 1D contiguous array in GPU
    // flat_index(b,h,n,d) = (((b * H) + h) * N + n) * D + d

    // Compute base offset for this (batch, head) block
    // 1 batch has num_heads heads
    const long base_offset = (static_cast<long>(batch_idx) * num_heads + head_idx) * static_cast<long>(seq_len) * head_dim; 
    // Locate the start of the query vector handled by this block
    const long query_offset = base_offset + static_cast<long>(query_idx) * head_dim; 

    // This block handles one query token for a specific (batch, head)
    // Compute pointer to the query vector in global memory
    const T* query_ptr = q_ptr + query_offset;  // Q[b,h,query_idx,0]
    // Compute pointer to the start of "all" keys for this (batch, head)
    const T* key_head_ptr = k_ptr + base_offset; // K[b,h,0,0]
    // Compute pointer to the start of "all" values for this (batch, head)
    const T* value_head_ptr = v_ptr + base_offset; // V[b,h,0,0]

    // Compute pointer to the output buffer for this (batch, head)
    T* output_head_ptr = output_ptr + base_offset; 
    // Compute how many key positions are visited under the stride pattern
    const int max_keys = (seq_len + stride - 1) / stride; 
    // Precompute the scaling factor applied to dot products
    const acc_t scale = reciprocal_sqrt_int<acc_t>(head_dim); 

    // Here we have computed the logical coordinate of q, k, v pointers and can use them to access the memory
    // ======================================================
    // Check whether this thread participates in loading a query component
    
    // * Each CUDA block has, say, blockDim.x = 256 threads, but our head_dim (the embedding size per attention head) might be smaller (e.g., 64, 128, or 192).
    // * We only need head_dim threads to load the actual query vector.
    // * So this condition ensures:
    //     1. Only the first head_dim threads read real data from global memory (query_ptr).
    //     2. The remaining threads (if any) safely write 0 — so that later reductions don't read garbage (uninitialized) shared memory values.
    
    if (thread_idx < head_dim) { 
        // Write the assigned query component into shared memory
        shared_query[thread_idx] = static_cast<acc_t>(query_ptr[thread_idx]); 
    } else { // Handle threads that exceed the head dimension
        // Store zero to avoid undefined values during reductions
        shared_query[thread_idx] = acc_t(0); 
    }
    __syncthreads(); // Ensure the entire query vector is available to all threads (join threads)

    const int warp_size = warpSize; // Cache warp size for distributing work
    const int warp_id = thread_idx / warp_size; // Warp identifier inside the block
    const int lane_id = thread_idx % warp_size; // Lane identifier within the warp
    const int warps_per_block = (blockDim.x + warp_size - 1) / warp_size; // Total warps participating in the block

    // Iterate over stride-aligned key positions and assign one warp per key to maximize utilization
    for (int key_base = 0; key_base < max_keys; key_base += warps_per_block) { 
        const int key_slot = key_base + warp_id; // Key slot handled by the current warp
        const bool warp_active = key_slot < max_keys; // Only some warps may have valid work in the final tile
        const int key_idx = key_slot * stride; // Translate slot index into sequence position
        const bool valid_key = warp_active && (key_idx < seq_len); // Mask keys that fall past sequence length

        acc_t partial_dot = acc_t(0); // Each lane accumulates its slice of the dot product
        if (valid_key) {
            const T* key_ptr = key_head_ptr + static_cast<long>(key_idx) * head_dim; // Pointer to the selected key vector
            for (int dim = lane_id; dim < head_dim; dim += warp_size) { 
                partial_dot += shared_query[dim] * static_cast<acc_t>(key_ptr[dim]); // Accumulate per-lane contributions
            }
        }

        const acc_t dot = warp_reduce_sum(partial_dot); // Reduce within the warp to form the full dot product
        if (lane_id == 0 && warp_active) { // One thread per warp commits the scaled score
            shared_scores[key_slot] = valid_key ? dot * scale : -std::numeric_limits<acc_t>::infinity();
        }
    }
    __syncthreads(); // Make sure all attention scores are written before normalization

    // Initialize the running maximum used for softmax normalization
    acc_t local_max = -std::numeric_limits<acc_t>::infinity(); 
    // Let each thread inspect a subset of scores
    for (int slot = thread_idx; slot < max_keys; slot += blockDim.x) { 
        // Update the thread-local maximum with observed scores
        local_max = local_max > shared_scores[slot] ? local_max : shared_scores[slot]; 
    }
    shared_buffer[thread_idx] = local_max; // Store each thread's local maximum for reduction
    __syncthreads(); // Synchronize before the reduction begins

    // Reduce to find the global maximum score (tree reduction)
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) { 
        if (thread_idx < offset) { // Restrict work to active threads per stage
            const acc_t other = shared_buffer[thread_idx + offset]; // Fetch the competitor maximum from the paired thread
            shared_buffer[thread_idx] = shared_buffer[thread_idx] > other ? shared_buffer[thread_idx] : other; // Keep the larger value between the pair
        }
        __syncthreads(); // Synchronize before continuing the reduction
    }
    const acc_t max_score = shared_buffer[0]; // Broadcast the maximal score within the block

    acc_t local_sum = acc_t(0); // Initialize running sum for exponentiated scores
    // Iterate over assigned stride slots
    for (int slot = thread_idx; slot < max_keys; slot += blockDim.x) { 
        const acc_t shifted = shared_scores[slot] - max_score; // Shift score by the maximum for numerical stability
        const acc_t expo = exp(shifted); // Compute exponential of the shifted score
        shared_scores[slot] = expo; // Overwrite the slot with the unnormalized attention weight
        local_sum += expo; // Accumulate the partial sum of exponentials
    }
    shared_buffer[thread_idx] = local_sum; // Store partial sums for reduction
    __syncthreads(); // Synchronize before summing across threads

    // Perform reduction to obtain the full softmax denominator
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) { 
        if (thread_idx < offset) { // Keep active threads inside reduction stages
            shared_buffer[thread_idx] += shared_buffer[thread_idx + offset]; // Accumulate upper half into lower half
        }
        __syncthreads(); // Synchronize before next reduction iteration
    }
    if (thread_idx == 0) { // Allow a single thread to store the completed denominator
        shared_denom = shared_buffer[0] + acc_t(1e-6); // Save the softmax denominator including a small epsilon for stability
    }
    __syncthreads(); // Ensure every thread sees the finalized denominator

    // Convert exponentials into probabilities
    for (int slot = thread_idx; slot < max_keys; slot += blockDim.x) { 
        shared_scores[slot] = shared_scores[slot] / shared_denom; // Normalize each stride weight by the denominator
    }
    __syncthreads(); // Synchronize before using probabilities to weight values

    if (thread_idx < head_dim) { // Assign output computation to threads mapping to head dimensions
        acc_t weighted_sum = acc_t(0); // Initialize the accumulator for the weighted value component

        // Traverse every stride slot to accumulate contributions
        for (int key_slot = 0; key_slot < max_keys; ++key_slot) { 
            const int key_idx = key_slot * stride; // Map slot to actual key position
            if (key_idx >= seq_len) { // Skip keys beyond the valid sequence range
                continue; // Ignore contributions from padded slots
            }

            // Fetch the computed attention probability for this key
            const acc_t weight = shared_scores[key_slot];
            // Determine the index inside the value tensor
            const long value_offset = static_cast<long>(key_idx) * head_dim + thread_idx; 
            // Accumulate weighted contribution from value component
            weighted_sum += weight * static_cast<acc_t>(value_head_ptr[value_offset]); 
        }
        output_head_ptr[static_cast<long>(query_idx) * head_dim + thread_idx] = static_cast<T>(weighted_sum); // Write the finished output component back to global memory
    }
}


// C++ function that dispatches the CUDA kernel
// CPU Code
torch::Tensor strided_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int stride) {

    // Validate inputs
    TORCH_CHECK(q.is_cuda(), "Input tensor Q must be on a CUDA device");
    TORCH_CHECK(q.is_contiguous(), "Input tensor Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "Input tensor K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "Input tensor V must be contiguous");

    const auto batch_size = q.size(0);
    const auto num_heads = q.size(1);
    const auto seq_len = q.size(2);
    const auto head_dim = q.size(3);

    // Create an output tensor of the same shape as Q
    auto output = torch::empty_like(q);
    
    // IMPORTANT FOR INDEXING
    // Grid  →  made of many Blocks
    // Block →  made of many Threads
    // Define grid and block dimensions
    // Grid: One block per (batch, head, query_token)
    dim3 gridDim(batch_size * num_heads, seq_len);
    // Block: Threads to parallelize the work for a single query token
    dim3 blockDim(256); // A common choice, can be tuned

    // Dispatch the kernel based on the data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "strided_attention_forward", ([&] {
        strided_attention_forward_kernel<scalar_t><<<gridDim, blockDim>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            stride
        );
    }));

    // Check for any CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}


// Bind the C++ function to a Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("strided_attention_forward", &strided_attention_forward_cuda, "Strided Attention Forward (CUDA)");
}
