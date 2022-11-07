# gpu-benchmarks

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
