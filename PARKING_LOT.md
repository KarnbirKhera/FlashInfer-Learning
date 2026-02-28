Shared Reduced Sum:
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

The following talks about 7 increasingly optimized versions of Shared Reduced Sum!
I was able to implement the first three, but this would be a lovely optimization opportunity.
>Credits to Mark Harris for this amazing insight!

- Atomic add can be used to avoid consecutive kernel launches.
Pro: Great for small n sizses
Con: For large n sizes, serialization can cause deficit in performance.
>Overall requires more experimentations to conclude behavior.



Matrix Transpose:
https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

A very helpful source!
> Also credits to Mark Harris for this amazing insight!


GEMM:
https://nerffer.github.io/matrixMultiplicationVisualization/
http://matrixmultiplication.xyz/

Both of the following links are great visualizations for matrix multiplication

-----------------------
Why is the naive uncoalesced access. At first I thought it was because in Matrix B we request for data column wise.
Although one could argue on a warp scale, Matrix A actually causes our uncoalesced access where on a warp scale, we are going down each row where
the addresses are not consecutive.

                *Disputed Self Theory, to be parked in "PARKING_LOT.MD" for after project submission understanding*
 At first I assumed Matrix B was the reason for this uncoalesced access, since on a thread level, we are requesting data from multiple rows.
 In reality on a warp scale, Matrix A is the reason why we may have uncoalesced access.
 This is because while a single thread in Matrix A travels across a row (coalesced), on a warp scale of 32 threads, we are actually 
 following the row dimension (threadIdx.y). This means:
   - Thread 0: Requests for address 0
   - Thread 1: Requests for address 33 because we are going down the Matrix A rows

---> Future perspective

Matrix A requests 32 bytes, but they are not used instantly, and have to wait for the next k iteration use the requested data.
This leaves a chance for the L1 or L2 to evict the data before the kernel has a chance to use it

Matrix B requests 32 bytes and is not reliant on k. It is reliant on the iterator value col. This col value is determined by threadIdx.x
which means all 32 threads can simutaniously use the bytes requested rather than needing to wait on iterator k.


--------------------




Interesting topics:

- When you come across a row-wise matrix, you must use row wise access form of base formula

Coordinate * Stride + Offset !!

This is because when the array is flattened (how the kernel sees it), row numbers are kept contingent.

If we used column wise access formula, we would get the wrong data!!

(This insight took many hours I fear)