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






Interesting topics:

- When you come across a row-wise matrix, you must use row wise access form of base formula

Coordinate * Stride + Offset !!

This is because when the array is flattened (how the kernel sees it), row numbers are kept contingent.

If we used column wise access formula, we would get the wrong data!!

(This insight took many hours I fear)