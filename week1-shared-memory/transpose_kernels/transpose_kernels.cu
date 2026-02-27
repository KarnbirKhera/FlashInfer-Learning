
#include <iostream>


/*
 Naive Transpose

- Reads are coalesced row wise
- Writes are not coalesced as they are column wise. This means the address between accesses are not consecutive.
- Uncoalesced writes results in implicit reads because of partial write mask in L2.
*/
__global__ void transpose_naive(float* input, float* output, size_t width, size_t height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < height && col < width) {
        //Writes are done down the row, resulting in uncoalesced writes. Inefficient as we only modify 4 bytes out of the 32 we request
        //from a DRAM sector. Confirmed by L2 GLobal Store Access Pattern in Nsight Compute
        output[col * height + row] = input[row * width + col];
    }
}








/* 
 Shared Transpose
  54.62% faster than Naive

- Writes are coalesced to DRAM
- Shared memory access is serialized due to threads hitting same bank

The reason shared transpose works so gracefully is when we read our shared memory data, we can tranpose our reads so that 
when we write to the DRAM, it can be done row wise. This makes our writes to global memory coalesced! What an amazing technique!

*/
#define tileSize 32
__global__ void transpose_shared(float* input, float* output, size_t width, size_t height) {
    // No padding, threads accessing the same column hit the same bank,
    // causing bank conflicts and serialization
    __shared__ float sharedMemory[tileSize][tileSize];


    // Which block are we at row wise * How many threads per row + Which thread in the row
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Which block are we in column wise * how many threads per column + Which thread in the column
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(col < width && row < height) {
        //Results in Shared Load Bank Conflicts
        sharedMemory[threadIdx.y][threadIdx.x] = input[row * width + col];
    }

    // Lets make sure all threads have completed before we move on.
    __syncthreads();

    //Transpose where 
    // col -> row
    // row -> col
    // Which block are we in column wise * how many threads per row + which thread in the row
    int transposed_col = blockIdx.y * blockDim.x + threadIdx.x;
    // Which block are we in row wise * how many threads per column +  which column in the row
    int transposed_row = blockIdx.x * blockDim.y + threadIdx.y;

    if(transposed_col < width && transposed_row < height) {
        //We write to our DRAM row wise, this results in a coalesced write. We modify all 32 bytes from the DRAM sector we request.
        //Confirmed by Nsight Compute as we do not face the same problem as the naive.
        output[transposed_row * height + transposed_col] = sharedMemory[threadIdx.x][threadIdx.y];
    }
}



/* 
 Shared Padded Transpose
  65.88% faster than Naive

- Writes are coalesced to DRAM
- Shared memory is accessed where each thread hits its respective bank
- Avoids bank conflicts by adding one to our data's row size. This allows row to map to its own bank when requesting!
*/
#define tileSize 32
__global__ void transpose_shared_padded(float* input, float* output, size_t width, size_t height) {
    //Avoids bank conflicts by adding one to our data's row size.
    /*
        Without: All rows % 32 results in hitting the same bank
        With: Rows hit their respective bank in the 32 banks of shared memory
    */
    __shared__ float sharedMemory[tileSize][tileSize+1];


    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(col < width && row < height) {
        sharedMemory[threadIdx.y][threadIdx.x] = input[row * width + col];
    }

    __syncthreads();


    int transposed_col = blockIdx.y * blockDim.x + threadIdx.x;
    int transposed_row = blockIdx.x * blockDim.y + threadIdx.y;

    if(transposed_col < width && transposed_row < height) {
        output[transposed_row * height + transposed_col] = sharedMemory[threadIdx.x][threadIdx.y];
    }
}




int main() {
    const int HEIGHT = 1024;
    const int WIDTH = 1024;
    const size_t size = WIDTH * HEIGHT * sizeof(float);

    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    for(int i = 0; i < WIDTH * HEIGHT; i++) {
        h_input[i] = float(i);
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);

    auto verify = [&](const char* name) {
        cudaDeviceSynchronize();
        cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
        bool correct = true;
        for(int row = 0; row < HEIGHT && correct; row++)
            for(int col = 0; col < WIDTH && correct; col++)
                if(h_input[row * WIDTH + col] != h_output[col * HEIGHT + row])
                    correct = false;
        printf("\n\n\n");
        printf("%s: %s\n", name, correct ? "CORRECT" : "INCORRECT");
        printf("\n\n\n");
        cudaMemset(d_output, 0, size);
    };

    transpose_naive<<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT);
    verify("transpose_naive");

    transpose_shared<<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT);
    verify("transpose_shared");

    transpose_shared_padded<<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT);
    verify("transpose_shared_padded");

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
}