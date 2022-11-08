#pragma once

template <typename T>
__global__ void shrink_vector(T *x, T *out, T shrink_rate, int n);

template <typename T>
__global__ void shift_shrink_vector(T *X_dev, T *Opt_shift, T *out_dev, T shrink_rate, int n, int pop);

template <typename T>
__global__ void shrink_vector(T *x, T *out, T shrink_rate, int n){
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int id_dim = tid % n;   // dimensions id
    int id_ind = tid / n;   // individual id
    int index  = id_ind*n + id_dim;
    if(tid < n){
        out[index] = shrink_rate*x[index];
    }
}

// shrink_rate = FUNC_BOUND/X_BOUND 
template <typename T>
__global__ void shift_shrink_vector(T *X_dev, T *Opt_shift, T *out_dev, T shrink_rate, int n, int pop){
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    
    if(tid < n*pop){
        // shift vector and then shrink
        out_dev[tid] = shrink_rate*(X_dev[tid] - Opt_shift[tid%n]);
    }
}
