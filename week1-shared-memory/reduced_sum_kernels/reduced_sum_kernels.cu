#include <iostream>


/**
 * NOTE: The following kernels only calculate sums on a block basis. If the host requires 5 blocks due to a large n amount,
 *       we will have 5 partial sums which we must sum once again by running the kernel again.
 */

/*
Reduced Sum Naive:

 - Prone to divergent branching because of "localId % (2 * stride) == 0".
   Active threads are spread across all warps, resulting in all warps requiring use.
*/
__global__ void reduced_sum_naive(float* input, float* output, int n) {
    // Which block row wise * number of threads row wise + which thread are we row wise
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Local thread for names sake
    int localId = threadIdx.x;

    // Rather than a fixed size, we can use extern where size is determined at launch time
    extern __shared__ float sharedMemory[];

    //Ensure thread has valid value. Incase threads in block are above # of elelments provided
    if(id < n) {
        sharedMemory[localId] = input[id];
    } else {
        //Ensure all threads have data to compute
        sharedMemory[localId] = 0;
    }
    __syncthreads();

    for(int stride = 1; stride < blockDim.x; stride *= 2) {
        //Warp divergence, only half the threads are doing work!
        if(localId % (2 * stride) == 0) {
            sharedMemory[localId] += sharedMemory[localId + stride];
        }
        __syncthreads();
    }

    if(localId == 0) {
        output[blockIdx.x] = sharedMemory[0];
    }

}


/*
 Reduced Sum Interweaved:
  35.96% faster than Naive

 - Prone to divergent branching because of "index + stride < blockDim.x"
   All active threads are clustered into required warps. Allowing entire warps to be inactive, resulting in decreased
   intra-warp divergence.

 - Prone to bank conflicts because of "int index = 2 * stride * localId;"
   Our shared memory access pattern is multiplicative where during larger strides many threads will hit the same bank
   resulting in serializing.
*/
__global__ void reduced_sum_interweaved(float* input, float* output, int n) {
    // Which block row wise * number of threads row wise + which thread are we row wise
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Local thread for names sake
    int localId = threadIdx.x;

    // Rather than a fixed size, we can use extern where size is determined at launch time
    extern __shared__ float sharedMemory[];

    //Ensure thread has valid value. Incase threads in block are above # of elelments provided
    if(id < n) {
        sharedMemory[localId] = input[id];
    } else {
        //Ensure all threads have data to compute
        sharedMemory[localId] = 0;
    }
    __syncthreads();

    for(int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * localId;
        if(index + stride < blockDim.x) {
            sharedMemory[index] += sharedMemory[index + stride];
        }
        __syncthreads();
    }

    if(localId == 0) {
        output[blockIdx.x] = sharedMemory[0];
    }

}


/*
 Reduced Sum Sequential
  38.42% faster than Naive

 - Prone to divergent branching because of "localId < stride"
   All active threads are clustered into required warps. Allowing entire warps to be inactive, resulting in decreased
   intra-warp divergence.

 - Solves share bank conflicts by using addition rather than multiplication where consecutive threads access
   consecutive shared memory banks.

*/
__global__ void reduced_sum_sequential(float* input, float* output, int n) {
    // Which block row wise * number of threads row wise + which thread are we row wise
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Local thread for names sake
    int localId = threadIdx.x;

    // Rather than a fixed size, we can use extern where size is determined at launch time
    extern __shared__ float sharedMemory[];

    //Ensure thread has valid value. Incase threads in block are above # of elelments provided
    if(id < n) {
        sharedMemory[localId] = input[id];
    } else {
        //Ensure all threads have data to compute
        sharedMemory[localId] = 0;
    }
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if(localId < stride) {
            sharedMemory[localId] += sharedMemory[localId + stride];
        }
        __syncthreads();
    }

    if(localId == 0) {
        output[blockIdx.x] = sharedMemory[0];
    }

}



/*
 Reduced Sum Sequential
  61.41% faster than Naive

 - Prone to divergent branching because of "localId < stride"
   All active threads are clustered into required warps. Allowing entire warps to be inactive, resulting in decreased
   intra-warp divergence.

 - Uses __shfl_down_sync when number of active threads is below 32. Rather than using shared memory for this last iteration,
   we can use the warp's registers to exchange data between threads!

*/
__global__ void reduced_sum_sequential_shfl_down(float* input, float* output, int n) {
    // Which block row wise * number of threads row wise + which thread are we row wise
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Local thread for names sake
    int localId = threadIdx.x;

    // Rather than a fixed size, we can use extern where size is determined at launch time
    extern __shared__ float sharedMemory[];

    //Ensure thread has valid value. Incase threads in block are above # of elelments provided
    if(id < n) {
        sharedMemory[localId] = input[id];
    } else {
        //Ensure all threads have data to compute
        sharedMemory[localId] = 0;
    }
    __syncthreads();


    for(int stride = blockDim.x/2; stride >= 32; stride >>= 1) {
        if(localId < stride) {
            sharedMemory[localId] += sharedMemory[localId + stride];
        }
        __syncthreads();
    }


        /*
        For the last 32 threads, we use a tournament style pass down of our values where:
        1st Iteration:
            Thread 0 -> get Thread 16 value
            Thread 1 -> get Thread 17 value
            Thread 2 -> get Thread 18 value
            ....
            Thread 16 -> get Thread 31 value
        
        2nd iteration:
            Thread 0 -> get Thread 8 value
            Thread 1 -> get Thread 9 value
            Thread 2 -> get Thread 10 value
            ...
            Thread 7 -> get Thread 15 value
        
        3rd iteration:
            Thread 0 -> get Thread 4 value
            Thread 1 -> get Thread 5 value
            Thread 2 -> get Thread 6 value
            Thread 3 -> get Thread 7 value
        
        4th iteration:
            Thread 0 -> get Thread 2 value
            Thread 1 -> get Thread 3 value
        5th iteration:
            Thread 0 -> get Thread 1 value
        After a simple 5 iterations, our Thread 0 holds the sum of all the values in the warp!
        This is much more faster than using shared memory.
    */
    if(localId < 32) {
        float sum = sharedMemory[localId];
        for(int offset = 16; offset > 0; offset >>= 1){
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if(localId == 0) {
            output[blockIdx.x] = sum;
        }
    }

}



int main() {
    const int N         = 1024 * 1024;
    const int blockSize = 256;
    const int gridSize  = (N + blockSize - 1) / blockSize;
    const size_t size   = N * sizeof(float);

    float* h_input   = (float*)malloc(size);
    float* h_partial = (float*)malloc(gridSize * sizeof(float));

    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    float cpu_sum = 0.0f;
    for (int i = 0; i < N; i++) cpu_sum += h_input[i];

    float *d_input, *d_output;
    cudaMalloc(&d_input,  size);
    cudaMalloc(&d_output, gridSize * sizeof(float));
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    size_t sharedMem = blockSize * sizeof(float);

    auto verify = [&](const char* name) {
        cudaDeviceSynchronize();
        cudaMemcpy(h_partial, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

        float gpu_sum = 0.0f;
        for (int i = 0; i < gridSize; i++) gpu_sum += h_partial[i];

        printf("\n\n\n");
        printf("%s: %s\n", name, (fabsf(gpu_sum - cpu_sum) < 1.0f) ? "CORRECT" : "INCORRECT");
        printf("\n\n\n");
        cudaMemset(d_output, 0, gridSize * sizeof(float));
    };

    reduced_sum_naive<<<gridSize, blockSize, sharedMem>>>(d_input, d_output, N);
    verify("reduced_sum_naive");

    reduced_sum_interweaved<<<gridSize, blockSize, sharedMem>>>(d_input, d_output, N);
    verify("reduced_sum_interweaved");

    reduced_sum_sequential<<<gridSize, blockSize, sharedMem>>>(d_input, d_output, N);
    verify("reduced_sum_sequential");

    reduced_sum_sequential_shfl_down<<<gridSize, blockSize, sharedMem>>>(d_input, d_output, N);
    verify("reduced_sum_sequential_shfl_down");

    free(h_input);
    free(h_partial);
    cudaFree(d_input);
    cudaFree(d_output);
}