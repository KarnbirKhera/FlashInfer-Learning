

/**
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
            // A[row * K + k] in matrix A, what row are we in, how many elements per row, which element are at in the row?
            // B[col * N + col] in matrix B, what column are we win, how many elemetns per column, what element are at in the column? SEE NOTE BELOW
            sum += A[row * K + k] + B[row * N + col]
        }
    }
}

/*
NOTE: One could ask the question, we use row * K + k for matrix A this makes sense, but why do we do 
      col * N + col for matrix B?

Answer: For matrix multiplication, we must tranpose matrix B, hence we have the following:
 Before Tranpose: col * K + k
 Af

*/