
#include <iostream>


/**                  -------- PRELUDE --------
 * I'll admit, it took a bit of visualization to understand how 
 * M, N and K relate to each other when doing matrix multiplication.
 * The following resource specifically was essential in my understanding.
 * 
 * http://matrixmultiplication.xyz/
 * 
 * From this source I have the following understanding.
 * Given two matrixes:
 * 
 *     A       B
 *  [3 x 2] [2 x 3]
 *   M * K   K * N 
 * 
 * Where M is the outer dimension of matrix A
 * Where N is the outer dimension of matrix B
 * Where K is the dimension both matrixes share
 * 
 * The process is:
 * 1. M * K
 * 2. K * N
 * 3. We do K number of sums (the values we multiplied along the shared dimension)
 * 
 * 
 * When we finish with our matrix multiplication, our result is in the outer dimension of M * N.
 *
 * 
 * Now that we know this, lets start!
 */


__global__ void gemm_naive(float* A, float* B, float* C, int M, int N, int K) {
    // Our current thread column wise within the block
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Our current thread row wise within block
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    //Ensure when we're going down our row wise matrix A, we dont go out of bounds (M)
    //Ensure when we're going left to right in our column wise matrix B, we dont go out of bounds (N)
    if(row < M && col < N) {
        float sum = 0.0f;
        
        //We run K many times (the shared dimension)
        for(int k = 0; k < K; ++k) {
            //See note below for further explanation
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

/*
NOTE:
When trying to grab the correct index for our matrix multiplication, I was very confused.
After a few hours of understanding (and a good nights rest), I was able to come up with the following.

The base formula for any time we're grabbing something from a matrix is derived from:

Coordinate * Stride + Offset

This is universal, we use it to find what our thread ID is in our current block in our grid.
It's used in neural networks for weight * input + bias.
And its used for finding the element we want to transpose as well.

The question is, given the context of naive GEMM where we have to matrixes that are row wise stored,
how can we use the formula to derive how to properly get the element we want?


Let's first describe the difference between a row wise matrix, and a column wise matrix.
Say we have the data 1,2,3,4,5,6. This is how the following data would be stored in our matrix.

            | 1 2 3 |
            | 4 5 6 |

When we flatten both of these matrixes to 1D, because thats how the computer sees them, we get the following two ways to store them:
Row Wise:
            | 1 2 3 4 5 6 |

Column Wise:
            | 1 4 2 5 3 6|

So despite having the same data, the way our data is stored influences how we are going to use Coordinate * Stride + Offset
to get the element we want. 


Now lets apply this to the context of naive GEMM where both matrixes are row wise. We start off with the formula 
Coordinate * Stride + Offset

When applied to the context of a row wise matrix we get:
(What row are we in) * (How big is the row) + (Where in the row are we)

Which can be translated to:
(Row) * (Row Size) + (Current index within row)


Now, looking at our GEMM code we have the following passed in variables:
A -> Matrix A location
B -> Matrix B location
C -> Matrix C location

M -> Outer dimension of matrix A (size)
N -> Outer dimension of matrix B (size)
K -> Inner dimension of both matrixes that match (size)

And the work done in our kernel gives us:
col -> Current column of this thread
row -> Current row of this thread
k   -> Current index along the shared dimension
        (this index goes across columns in matrix A)
        (this index goes down rows in matrix B)

Now with the information we have, we can derive what equation we need for both matrix A and matrix B to get our corresponding element to matrix multiply

Matrix A, we're going across a row left to right:
    (What row are we in) * (How big is the row) + (Where in the row are we)
        ->
        A[row * K + k]

Matrix B, we're going down a column, but still must heed to our row wise matrix:

    (what row are we in) * (How big is the row) + (where in the row are we)
        ->
        B[k * N + col]


These two applications of our base formula Coordinate * Stride + Offset allows us to grab the index for both Matrix A and Matrix B!

/**
 * ------------------------------------------------------------------
 *       Matrix A                              Matrix B
 *        M x K                                 K x N
 * ------------------------------------------------------------------
 *  Which row * rowsize + col             Which row * rowsize + col
 * 
 *            K                                     N
 *   _  .-------------.                       _  .-------------.
 *  |   |====[k]===>==| ] row                |   |^|-----------| ] row
 *  |   |-------------|                      |   |^|-----------|
 * M|   |-------------|                     K|   |k|-----------|
 *  |   |-------------|                      |   |^|-----------|
 *  |_  '-------------'                      |_  '-------------'
 *      |_|                                      |_|
 *    col                                        col
 *
 *        A[row * K + k]                          B[k * N + col]
 * ------------------------------------------------------------------
 */

#define TILE_SIZE 32
__global__ void gemm_tiled(float* A, float* B, float* C, int M, int N, int K) {

    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    //Ceiling Integer Divison
    int numOfTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for(int i = 0; i < numOfTiles; ++i) {
        
        int aCol = i * TILE_SIZE + threadIdx.x;
        if(row < M && aCol < K) {
            A_shared[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else {
            A_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int bRow = i * TILE_SIZE + threadIdx.y;
        if(bRow < K && col < N) {
            B_shared[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            B_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for(int k = 0; k < TILE_SIZE; k++) {
            sum += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
        }

        __syncthreads();
    }

    if(row < M && col < N) {
        C[row * N + col] = sum;
    }
}




#define TILE_SIZE 32

 int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    const size_t sizeA = M * K * sizeof(float);
    const size_t sizeB = K * N * sizeof(float);
    const size_t sizeC = M * N * sizeof(float);

    float* hA = (float*)malloc(sizeA);
    float* hB = (float*)malloc(sizeB);
    float* hC_ref = (float*)malloc(sizeC);
    float* hC_gpu = (float*)malloc(sizeC);

    srand(42);
    for (int i = 0; i < M * K; i++) hA[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < K * N; i++) hB[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    // CPU reference
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += hA[i * K + k] * hB[k * N + j];
            hC_ref[i * N + j] = sum;
        }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);
    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    auto verify = [&](const char* name) {
        cudaDeviceSynchronize();
        cudaMemcpy(hC_gpu, dC, sizeC, cudaMemcpyDeviceToHost);

        float max_err = 0.0f;
        for (int i = 0; i < M * N; i++) {
            float err = fabsf(hC_ref[i] - hC_gpu[i]);
            if (err > max_err) max_err = err;
        }

        printf("\n\n\n");
        printf("%s: %s (max error: %.2e)\n", name, (max_err < 1e-3f) ? "CORRECT" : "INCORRECT", max_err);
        printf("\n\n\n");
        cudaMemset(dC, 0, sizeC);
    };

    gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    verify("gemm_naive");

    gemm_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    verify("gemm_tiled");


    free(hA);
    free(hB);
    free(hC_ref);
    free(hC_gpu);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}