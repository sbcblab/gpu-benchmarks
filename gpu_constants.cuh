#pragma once

#define WARP_SIZE       32 
#define MIN_OCCUPANCY   512 // threads per block
#define MAX_BLOCK_SIZE  1024 
#define CONST_MEM_SIZE  65536

#define WARPS_COUNT(x) ((x + (WARP_SIZE - 1)) / WARP_SIZE)

template <typename T>
__device__ void reduction_mult( int index, T *s_mem ){
    int i;
    for( i = blockDim.x / 2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            s_mem[index] *= s_mem[index + i];
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void reduction( int index, T *s_mem ){
    int i;

    for( i = blockDim.x / 2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            s_mem[index] += s_mem[index + i];
        }
        __syncthreads();
    }
    
}
