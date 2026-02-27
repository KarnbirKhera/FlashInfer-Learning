

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
Say we have the data 1,2,3,4,5,6. This is how the following data would be stored in both row wise vs column wise:

Row wise:
            | 1 2 3 |
            | 4 5 6 |

Column wise: 
            | 1 3 5 |
            | 2 4 6 |

When we flatten both of these matrixes to 1D, because thats how the computer sees them, we get the following:

Row Wise:
            | 1 2 3 4 5 6 |

Column Wise:
            | 1 3 5 2 4 6 |

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
        A[row * K * k]

Matrix B, we're going down a column, but still must heed to our row wise matrix:

    (what row are we in) * (How big is the row) + (where in the row are we)
        ->
        B[k * N + col]
*/
