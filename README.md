# gpu-benchmarks

This C++ library provides GPU-accelerated benchmark functions of the CEC 2022 bound-constrained continuous optimization suite. For using that, you can use this repository as a submodule of your project. Currently, the library has 12 functions:

| Nº |      Function      |  
|----------|:-------------:|
| 1 |  Zakharov Function |
| 2 |    Rosenbrock Function |
| 3 | Expanded Schaffer's _f6_ Function |
| 4 | Non-Continuous Rastrigin's Function |
| 5 | Levy Function |
| 6 | Hybrid Function 1 ( N = 3 ) |
| 7 | Hybrid Function 2 ( N = 6 ) |
| 8 | Hybrid Function 3 ( N = 5 ) |
| 9 | Composition Function 1 |
| 10 | Composition Function 2 |
| 11 | Composition Function 3 |
| 12 | Composition Function 4 |

## Requisites

To compile the library, you need to have CUDA installed on your system. Ensure that you have CUDA installed before proceeding with the compilation process.

Additionally, when compiling your code, it is necessary to link the cuBLAS library. To do this, include the `-lcublas` flag during compilation.

## Include

To use the library in your code, you should include the umbrella file `cuda_benchmark_suite.cuh` as desired. The file can be found at the following path: `/gpu-benchmarks/include`. We recommend adding this path to the search path by compiling your code with the `-I gpu-benchmarks/include` flag. Once you have done that, you can include the library using the following statement:
```
#include <cuda_benchmark_suite.cuh>
```

## Auxiliary Files

The benchmark functions rely on auxiliary files that contain rotation, shuffling, and shifting vectors. Currently, these functions expect the data to be stored in **binary** files as follows:

- Rotation Matrix: A single- or double-precision floating-point matrix of size $nxn$.
- Shift Array: A single- or double-precision floating-point array of size $n$.
- Shuffle Array: An integer array of size $n$.

To help you create these files and customize the benchmark according to your desired problem size, we provide a script in the `input_data` directory. For detailed instructions on how to use this script, please refer to the README file included in the `input_data` directory.

## Example 

An example of instantiation of a benchmark function looks like:
```
char shuffleFile[] =  "shufflefile.bin";
char shiftFile[] =  "shiftfile.bin";
char matrixFile[] =  "matrixfile.bin";
Benchmark<float> *B = createBenchmark<float>(DIM_SIZE, NP, FUNC_ID, shuffleFile, shiftFile, matrixFile);
```
