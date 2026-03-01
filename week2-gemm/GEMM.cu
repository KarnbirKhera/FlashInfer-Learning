
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


/* 
 GEMM Naive

- Global Load Access Pattern
  > 26.4 out of the 32 bytes transmitted per sector are utilized per thread
     - We are not using all 32 bytes of the DRAM sector we requested.
         >Matrix A requests 32 bytes, but they are not used instantly, and have to wait for the next k iteration use the requested data.
          This leaves a chance for the L1 or L2 to evict the data before the kernel has a chance to use it
         >Matrix B requests 32 bytes and is not reliant on k. It is reliant on the iterator value col. This col value is determined by threadIdx.x
          which means all 32 threads can simutaniously use the bytes requested rather than needing to wait on iterator k.
*/
__global__ void gemm_naive(float* A, float* B, float* C, int M, int N, int K) {
    // Our current thread column wise within the block
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Our current thread row wise within block. Note this value stays the same per warp where every thread will have the same row value.
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    //Ensure when we're going down our row wise matrix A, we dont go out of bounds (M)
    //Ensure when we're going left to right in our column wise matrix B, we dont go out of bounds (N)
    if(row < M && col < N) {
        float sum = 0.0f;
        
        //We run K many times (the shared dimension)
        for(int k = 0; k < K; ++k) {
            //See note below for further explanation
            float A_value = A[row * K + k];
            float B_value = B[k * N + col];
            sum += A_value * B_value;

            // Commented and seperated out for further SASS analysis into uncoalesced access reasoning
            //sum += A[row * K + k] * B[k * N + col];
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




/* 
 GEMM Tiled
  29.64% faster than Naive


- Shared memory use allows for re-use of loaded data.
  > Observed by -9.37% decrease in Device Memory Load sectors (1425372 -> 1291852)
    - A notice a small reduction in our DRAM load because we still require the same amount of data, but
      our access is coalesced hence we are using all 32 bytes of every DRAM sector we call for.
        > In the Naive kernel on average we would only use 26.4 bytes of out the 32 bytes we requested from each sector.
  > Observed by -57.79% decrease in L2 Load Requests (5612387 -> 2368962)
  > Observed by -96.83% decrease in L1 Load Requests (67141632 -> 2129920)
  We see a significant reduction in the Load requests from both the L1 and L2 as we no longer explicitly require them once we load the tile
  into Shared Memory, whereas in the Naive kernel we would hope the L1 and L2 had the data we needed.

  > Observed +inf increase in Shared Memory
     - Shared Memory Store (0 -> 2097152) 
     - Shared Memory Load (0 -> 41943040)
       > This is just a theory but one could assume if our Load count is significantly above our Store count,
         our justification to use shared memory is correct. I say this because each one of these loads could have
         been a L1 hit, L2 hit or a DRAM request which is significantly slower, but our use of Shared Memory allowed
         prevented this. 
          - The ratio between Load and Store is
            > 41,943,040 / 2,097,152 = 20x data reuse that could have been slow L1, L2 or DRAM requests!

- Shared memory use also solves the problem for Matrix A where the data requested may have been flushed from the L1 or L2 cache
  by the time threads need the data as they wait on the k iterator, see naive section for more details.
    > The problem is solved as Shared Memory is user controlled which allows us to keep the necessary data within fast reach.
*/

#define TILE_SIZE 32
__global__ void gemm_tiled(float* A, float* B, float* C, int M, int N, int K) {
/**************************** Intialize pointers & sum ***************************************************** */
    //Create Shared Memory for both matrices
    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

    //Get thread ID, usually we would do blockDim but the focus is on our current tile
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    //Ceiling Integer Divison
    int numOfTiles = (K + TILE_SIZE - 1) / TILE_SIZE;


    for(int i = 0; i < numOfTiles; ++i) {
/****************************Load tile A and tile B into Shared Memory ******************************************************/

        // i allows us to slide across K 
        // TILE_SIZE determines the size of the tile
        // threadIdx.x represents all 32 threads within our warp, meaning we are able to parallelize
        //  any computation derived from threadIdx.x
        int aCol = i * TILE_SIZE + threadIdx.x;
        if(row < M && aCol < K) {
            // We do [ThreadIdx.y][ThreadIdx.x] because Matrix A is stored row wise
            // Matching the row wise [fixed][changes] notation
            A_shared[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else {
            A_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // i allows us to slide across K
        // TILE_SIZE determines the size of the tile
        // threadIdx.y determines the row, threadIdx.y stays the same value across the warp.
        int bRow = i * TILE_SIZE + threadIdx.y;
        if(bRow < K && col < N) {
            // Store current element in shared memory
            // bRow = Our current row
            // N = Size of our row
            // col = Current value, derived from threadIdx.x and is parallelized
            B_shared[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            B_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        //Ensure all threads have completed, and data is ready
        __syncthreads();


/****************************Compute DOT product freely in Shared Memory ***************************************************** */
        //DOT product from shared memory
        for(int k = 0; k < TILE_SIZE; k++) {
            sum += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
        }

        //Ensure summing is complete
        __syncthreads();
    }
/**************************** Write sum to C ***************************************************** */
    //Write sum to Matrix C
    if(row < M && col < N) {
        C[row * N + col] = sum;
    }
}






#define TILE_SIZE 32
#define THREAD_ITEMS 4

__global__ void gemm_register_22d(float* A, float* B, float* C, int M, int N, int K) {

    __shared__ float A_shared[TILE_SIZE][TILE_SIZE+1];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

    // Each thread owns a 4x4 block of C
    int rowBase = blockIdx.y * TILE_SIZE + threadIdx.y * THREAD_ITEMS;
    int colBase = blockIdx.x * TILE_SIZE + threadIdx.x * THREAD_ITEMS;

    // 16 accumulators in registers
    float sum[THREAD_ITEMS][THREAD_ITEMS] = {{0.0f}};

    int numOfTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Loading setup: 64 threads, 1024 elements per tile, 16 loads each
    int linearIdx = threadIdx.y * blockDim.x + threadIdx.x;
    int loadPerThread = (TILE_SIZE * TILE_SIZE) / (blockDim.x * blockDim.y);

    for (int i = 0; i < numOfTiles; i++) {

        // Collaborative load: each thread loads 16 elements of A and B
        for (int l = 0; l < loadPerThread; i++) {
            int idx = l * (blockDim.x * blockDim.y) + linearIdx;
            int sRow = idx / TILE_SIZE;
            int sCol = idx % TILE_SIZE;

            int aRow = blockIdx.y * TILE_SIZE + sRow;
            int aCol = i * TILE_SIZE + sCol;
            if (aRow < M && aCol < K)
                A_shared[sRow][sCol] = A[aRow * K + aCol];
            else
                A_shared[sRow][sCol] = 0.0f;

            int bRow = i * TILE_SIZE + sRow;
            int bCol = blockIdx.x * TILE_SIZE + sCol;
            if (bRow < K && bCol < N)
                B_shared[sRow][sCol] = B[bRow * N + bCol];
            else
                B_shared[sRow][sCol] = 0.0f;
        }

        __syncthreads();

        // Compute: outer product of register fragments
        for (int k = 0; k < TILE_SIZE; ++k) {
            float a[THREAD_ITEMS];
            float b[THREAD_ITEMS];

            for (int ri = 0; ri < THREAD_ITEMS; ++ri)
                a[ri] = A_shared[threadIdx.y * THREAD_ITEMS + ri][k];

            for (int ci = 0; ci < THREAD_ITEMS; ++ci)
                b[ci] = B_shared[k][threadIdx.x * THREAD_ITEMS + ci];

            for (int ri = 0; ri < THREAD_ITEMS; ++ri)
                for (int ci = 0; ci < THREAD_ITEMS; ++ci)
                    sum[ri][ci] += a[ri] * b[ci];
        }

        __syncthreads();
    }

    // Write 4x4 results to C
    for (int ri = 0; ri < THREAD_ITEMS; ++ri) {
        for (int ci = 0; ci < THREAD_ITEMS; ++ci) {
            int r = rowBase + ri;
            int c = colBase + ci;
            if (r < M && c < N)
                C[r * N + c] = sum[ri][ci];
        }
    }
}





/*
    2D is geniunely troubling, requires additonal framework

    - Define dimensions for Global, Shared and Register dimension
    - Who am I? (our pointer/index what does it track)
    - Define workload (before loop, calcualt ehow much manual labor each thread has to do)
*/

/* 
    GRID DIMENSION: 1024x1024 (Passed into the function)
        - blockIdx.x > Where are we in the grid horizontally 
        - blockIdx.y > Where are we in the grid vertically
        - gridDim.x  > How big is the block horizontally
    BLOCK DIMENSION: 8x8
        - threadIdx.x > Where are we in the block horizontally 
        - threadIdx.y > Where are we in the block vertically
        - blockDim.x  > How big is the block horizontally
    SHARED DIMENSION: 32x32
        - TILE_SIZE > How big shared memory is horizontally/vertically
    REGISTER DIMENSION: 4x4
        - WORK_PER_THREAD > How much work each thread does/the stride


 FORMULA DERIVATION FOR THIS USE CASE:
    Base formula: Coordinate * Stride + Offset
        
        translates to:

    In human nomenclature: WhereAmI * HowBigAmI + WhereAmIWithinIt?

        translates to:

    Matrix is in row form hence: WhichRow * HowBigIsTheRow + WhereInTheRowAmI?


----
OH MY GOD EVERYTHING IS TECHNICALLY FLATTENED
THE  TWO TREE FRAMEWORK IS FANTASTIC!!!

*/

#define TILE_SIZE 32
#define WORK_PER_THREAD 4
__global__ void gemm_register_2d(float* A, float* B, float* C, int M, int N, int K) {
    
    /*
        Shell and Allocations
    */
    __shared__ float A_shared[32][33]; // [ELEMENT]
    __shared__ float B_shared[32][32]; // [ELEMENT]

    //Accumlate 4x4 in registers, avoid Shared Memory
    float accumulate[WORK_PER_THREAD][WORK_PER_THREAD] = {{0.0f}}; // [ELEMENT]

    //Thread mapping
    //TODO: Redo the dimensional analysis on these parts
    int col = blockIdx.x * blockDim.x + threadIdx.x * WORK_PER_THREAD; // Block * Thread + Thread = [Thread] 0-7
    int row = blockIdx.y * blockDim.y + threadIdx.y * WORK_PER_THREAD; // Block * Thread + Thread = [Thread] 0-7
    int flattenedId = threadIdx.y * blockDim.x + threadIdx.x; // Thread * Thread + Thread = [Thread] 0 - 63
    
    int numOfTiles = K / TILE_SIZE; // 32 
    int numOfStrides = (TILE_SIZE * TILE_SIZE) / (blockDim.x * blockDim.y); // 16

    // SLIDING LOOP - We move along the K dimension
    for(int tileId = 0; tileId < numOfTiles; tileId++) {

        //Task One: Load to Share Memory
        
        // STAMPING LOOP - We cover 32x32 area
        for(int stride = 0; stride < numOfStrides; stride++) {
            // 1. Flatten to flat thread count 0 - 63
            // 2. Expand to Shared Memory flat thread count 0 - 1023
            // 3. Fold to Shared Memory Dimension (32x32)
            // 4. Matrix A & B to Shared Memory
            
            // 2. 
            //Stored row wise hence formula is: Current Row * Size of Row + column 
            int sharedMemoryId = stride * (blockDim.x * blockDim.y) + flattenedId; // Max value check: 15 * (8 * 8) + 63 = 1023, matches 0-1023 expected

            //3.
            int sRow = sharedMemoryId / 32; // [ELEMENT]
            int sCol = sharedMemoryId % 32; // [ELEMENT]

            // 4.
            // Matrix A
            //  Must be X Axis Access. Where * Size + WhereIn
            int aCol = tileId * TILE_SIZE + sCol; // [BLOCK] * [ELEMENT/BLOCK] + [ELEMENT] = [ELEMENT], 0 - 1023
            int aRow = blockIdx.y * TILE_SIZE + sRow; // [BLOCK] * [ELEMENT/BLOCK] + [ELEMENT] = [ELEMENT], 0 - 1023
            if(aCol < K && aRow < M) {

            } else {

            }
            


        }
    }

}








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

    dim3 block_reg(TILE_SIZE / THREAD_ITEMS, TILE_SIZE / THREAD_ITEMS);  // (8, 8) = 64 threads

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

    gemm_register_2d<<<grid, block_reg>>>(dA, dB, dC, M, N, K);
    verify("gemm_register_2d");

    free(hA);
    free(hB);
    free(hC_ref);
    free(hC_gpu);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}