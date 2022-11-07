#include <cooperative_groups.h>
#include <stdio.h>
#include "gpu_constants.cuh"

namespace cg = cooperative_groups;


__global__ void shrink_vector(double *x, double *out, double shrink_rate, int n){
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int id_dim = tid % n;   // dimensions id
    int id_ind = tid / n;   // individual id
    int index  = id_ind*n + id_dim;
    if(tid < n){
        out[index] = shrink_rate*x[index];
    }
}

// shrink_rate = FUNC_BOUND/X_BOUND
__global__ void shift_shrink_vector(double *X_dev, double *Opt_shift, double *out_dev, double shrink_rate, int n, int pop){
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    // int id_dim = tid % n;   // dimensions id
    // int id_ind = tid / n;   // individual id:
    // int index  = id_ind*n + id_dim;
    if(tid < n*pop){
        // shift vector and then shrink
        out_dev[tid] = shrink_rate*(X_dev[tid] - Opt_shift[tid%n]);
    }
}


__global__ void mul_matrix_vector(double *M, double *V, double *out, int n){
    int row = blockIdx.x;

    double value = 0;

    extern __shared__ double mem[];
    
    for(int t = threadIdx.x; t < n; t += blockDim.x){
        value += V[t]*M[row*n + t];
    }

    mem[threadIdx.x] = value;
    __syncthreads();

    // reduction using the available threads in the block
    for(int i = blockDim.x/2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            mem[threadIdx.x] += mem[threadIdx.x + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        out[blockIdx.x] = mem[0];
    }
}


// using 4 warps per index of the final vector
// this kernel must be launched with 128 threads per block
__global__ void mul_matrix_vector_2(double *M, double *V, double *out, int n){ 
    int row = blockIdx.x;    
    __shared__ double smem[4];

    cg::thread_block_tile<WARP_SIZE> group = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

    double value = 0; 

    for(int t = threadIdx.x; t < n; t += blockDim.x){    
        value += M[row*n + t]*V[t]; 
    }

    value += group.shfl_down(value, WARP_SIZE/2);
    value += group.shfl_down(value, WARP_SIZE/4);
    value += group.shfl_down(value, WARP_SIZE/8);
    value += group.shfl_down(value, WARP_SIZE/16);
    value += group.shfl_down(value, WARP_SIZE/32);

    // TODO: try to group using coalesced threads and use a reduction
    if(group.thread_rank() == 0){
        smem[threadIdx.x / WARP_SIZE ] = value;
    }
    __syncthreads();
    
    if(threadIdx.x < 4){
        value = smem[threadIdx.x];
    }

    value += group.shfl_down(value, 2);
    value += group.shfl_down(value, 1);

    if(threadIdx.x == 0){
        out[row] = value;
    }

}


__global__ void mul_matrix_vector_3(double *M, double *V, double *out, int n){
    int row = blockIdx.x;

    double value = 0;

    extern __shared__ double mem[];

    for(int t = threadIdx.x; t < n; t += blockDim.x){
        value += V[t]*M[row*n + t];
    }

    mem[threadIdx.x] = value;
    __syncthreads();

    // reduction using the available threads in the block
    for(int i = blockDim.x/2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            mem[threadIdx.x] += mem[threadIdx.x + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        out[blockIdx.x] = mem[threadIdx.x];
    }
}