
CC =nvcc
CFLAGS =
GPUFLAGS=-arch=sm_52 -lcublas -lm
DIR=build
OUT_FILE=out.a

all: main

main: main.cu aux_main.cpp
	$(CC) -o $(DIR)/$(OUT_FILE) $^ $(CFLAGS) $(GPUFLAGS)

clean: 
	rm *.o $(DIR)/$(OUT_FILE)

