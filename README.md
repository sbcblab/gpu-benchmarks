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

For compiling the library you need CUDA installed.   

## include
The target must be compiled with the flag to include the path to the library's umbrella file `-Igpu-benchmarks/include`, and the flag to include cuBLAS `-lcublas`. Then, include the library using:
```
#include <cuda_benchmark_suite.cuh>
```
An example of instantiation of a benchmark function looks like:
```
char shuffleFile[] =  "shufflefile.bin";
char shiftFile[] =  "shiftfile.bin";
char matrixFile[] =  "matrixfile.bin";
Benchmark<float> *B = createBenchmark<float>(DIM_SIZE, NP, FUNC_ID, shuffleFile, shiftFile, matrixFile);
```

## build

From the `gpu-benchmarks` directory:
```
make
```
