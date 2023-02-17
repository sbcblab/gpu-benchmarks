# gpu-benchmarks
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
## run
The main function is an example on how to use the benchmarks. For now, it is mandatory for the benchmark functions to use rotation matrices and shift vectors. 
```
./build/out.a -d DIMENSIONS -p POPULATION_SIZE -f FUNCTION_NUMBER -i INDIVIDUALS_PER_BLOCK  
```
