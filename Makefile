
CC =nvcc
CFLAGS =
GPUFLAGS=-arch=sm_52 -lcublas
DIR=build
OUT_FILE=out.a

all: main

main: main.cu benchmark_kernels.o utils.o vector_ops.o
	$(CC) -o $(DIR)/$(OUT_FILE) $^ $(CFLAGS) $(GPUFLAGS)

objects: 
	$(CC) -c utils.cpp $(GPUFLAGS)

main.o: tests.cu
	$(CC) -c $^ $(CFLAGS)  $(GPUFLAGS)

benchmark_kernels.o: benchmark_kernels.cu  
	$(CC) -c $^ $(CFLAGS) $(GPUFLAGS)

vector_ops.o: vector_ops.cu
	$(CC) -c $^ $(CFLAGS) $(GPUFLAGS)


clean: 
	rm *.o $(DIR)/$(OUT_FILE)

