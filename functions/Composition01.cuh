
#pragma once

#include <stdio.h>

#include "Benchmark.cuh"

#ifndef ROSENBROCK_KERNEL
template <typename T>
__global__ void rosenbrock_gpu(T *x, T *f, int nx);
#endif

#ifndef ELLIPS_KERNEL
template <typename T>
__global__ void ellips_gpu(T *x, T *f, int nx);
#endif

#ifndef BENTCIGAR_KERNEL
template <typename T>
__global__ void bent_cigar_gpu(T *x, T *f, int nx);
#endif

#ifndef DISCUS_KERNEL
template <typename T>
__global__ void discus_gpu(T *x, T *f, int nx);
#endif

template <class T> 
class Composition01 : public Benchmark<T> {
    private:
        int cf_num = 5;
        T * p_cfit_dev;
        T * p_tcfit_dev;

        void allocateMemory(){
            cudaMalloc<T>(&(this->p_aux_dev), (this->n)*(this->pop_size)*sizeof(T));
            cudaMalloc<T>(&p_tcfit_dev, cf_num*(this->pop_size)*sizeof(T));
            cudaMalloc<T>(&p_cfit_dev, cf_num*(this->pop_size)*sizeof(T));
        }

        void freeMemory(){
            cudaFree(this->p_aux_dev);
            cudaFree(p_cfit_dev);
            cudaFree(p_tcfit_dev);

            cublasDestroy(this->handle);
            
            if(this->shift_func) cudaFree(this->p_shift_dev);
            if(this->rot_func)   cudaFree(this->p_rotm_dev);

            this->freeIO();
        }


        void transpose_fit() { }

    public:
        
        Composition01(int _n, int _pop_size){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);
            
            cublasCreate(&(this->handle));

            allocateMemory();

            float delta[5]  {10, 20, 30, 40, 50};
            float bias[5]  = {0, 200, 300, 100, 400};
            float lambda[5] = {10000/1e+4, 10000/1e+10, 10000/1e+10, 10000/1e+10, 10000/1e+10};

            // memcpys may ovewrite useful data in constant memory if multiple composition functions are instantiated
            cudaMemcpyToSymbol(p_delta_dev, delta, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(p_bias_dev, bias, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(p_lambda_dev, lambda, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);

        }

        Composition01(int _n, int _pop_size, char shift_filename[], char matrix_filename[]){

            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);
            
            cublasCreate(&(this->handle));

            this->use_rotation_matrix(matrix_filename, _n*_n*cf_num);
            this->use_shift_vector(shift_filename, _n*cf_num);            

            allocateMemory();

            float delta[5]  {10, 20, 30, 40, 50};
            float bias[5]  = {0, 200, 300, 100, 400};
            float lambda[5] = {10000/1e+4, 10000/1e+10, 10000/1e+10, 10000/1e+10, 10000/1e+10};

            // memcpys may ovewrite useful data in constant memory if multiple composition functions are instantiated
            cudaMemcpyToSymbol(p_delta_dev, delta, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(p_bias_dev, bias, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(p_lambda_dev, lambda, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);

        }
        ~Composition01(){
            freeMemory();
        }


        void compute(T *p_x, T*p_f){
            
            this->checkPointers(p_x, p_f);
            
            int offset = 0;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, this->p_shift_dev, this->p_aux_dev, ROSENBROCK_BOUND/X_BOUND, this->n, this->pop_size);
            this->rotation(this->p_rotm_dev);
            rosenbrock_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->rot_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            
            offset = 1;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, &(this->p_shift_dev[offset*this->n]), this->p_aux_dev, ELLIPSIS_BOUND/X_BOUND, this->n, this->pop_size);
            this->rotation(&(this->p_rotm_dev[this->n*this->n]));
            ellips_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->rot_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            
            offset = 2;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, &(this->p_shift_dev[offset*(this->n)]), this->p_aux_dev, BENT_CIGAR_BOUND/X_BOUND, this->n, this->pop_size);
            this->rotation(&(this->p_rotm_dev[offset*this->n*this->n]));
            bent_cigar_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->rot_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            
            offset = 3;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, &(this->p_shift_dev[offset*(this->n)]), this->p_aux_dev, DISCUS_BOUND/X_BOUND, this->n, this->pop_size);
            this->rotation(&(this->p_rotm_dev[offset*this->n*this->n]));
            discus_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->rot_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            
            offset = 4;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, &(this->p_shift_dev[offset*(this->n)]), this->p_aux_dev, ELLIPSIS_BOUND/X_BOUND, this->n, this->pop_size);
            ellips_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->p_aux_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            
            transpose_fit();
            dim3 BLOCK(32, this->cf_num);

            cfcal_gpu<<<this->pop_size, BLOCK>>>(  this->p_x_dev, 
                                                    this->p_f_dev, 
                                                    this->p_shift_dev, 
                                                    this->p_tcfit_dev, 
                                                    this->n );
            this->checkOutput(p_f);
        }

};

template<> 
void Composition01<double>::transpose_fit(){
    double alpha = 1.0;
    double beta  = 0.0;

    cublasDgeam( this->handle, 
                 CUBLAS_OP_T, 
                 CUBLAS_OP_N, 
                 cf_num, 
                 this->pop_size, 
                 &alpha, 
                 this->p_cfit_dev, 
                 this->pop_size, 
                 &beta, 
                 NULL, 
                 this->pop_size, 
                 this->p_tcfit_dev, 
                 cf_num);
}

template<> 
void Composition01<float>::transpose_fit(){
    float alpha = 1.0;
    float beta  = 0.0;

    cublasSgeam( this->handle, 
                 CUBLAS_OP_T, 
                 CUBLAS_OP_N, 
                 cf_num, 
                 this->pop_size, 
                 &alpha, 
                 this->p_cfit_dev, 
                 this->pop_size, 
                 &beta, 
                 NULL, 
                 this->pop_size, 
                 this->p_tcfit_dev, 
                 cf_num);
}

#ifndef ROSENBROCK_KERNEL
#define ROSENBROCK_KERNEL
template <typename T>
__global__ void rosenbrock_gpu(T *x, T *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;
    
    extern __shared__ T s_mem[];

    T xi = 0;
    T x_next = 0;
    T sum = 0;

    if(threadIdx.x < nx){
        xi = x[chromo_id*nx + threadIdx.x];
        s_mem[gene_block_id] = xi;
    }
    __syncthreads();

    if(threadIdx.x < nx){
        if(threadIdx.x < blockDim.x - 1 ){ // if it is not the last thread
            x_next = s_mem[gene_block_id + 1];
            sum = 100*(xi*xi - x_next)*(xi*xi- x_next) + (xi - 1)*(xi - 1);
        }
    }


    
    const int n_blockdims = (int)(blockDim.x*ceil((float)nx/blockDim.x));

    // utilizar um for loop que utilize todas as thread, e então um if i < nx 
    for(i = threadIdx.x + blockDim.x; i < n_blockdims; i += blockDim.x){
        if(i < nx){
            s_mem[gene_block_id] = x[chromo_id*nx + i];
        }
        __syncthreads();

        if(i < nx){
            if(threadIdx.x == blockDim.x - 1){  // if last thread, compute previous steps
                x_next = s_mem[gene_block_id - threadIdx.x];
                sum += 100*(xi*xi - x_next)*(xi*xi - x_next) + (xi - 1)*(xi - 1);
            }

            xi = s_mem[gene_block_id];

            if(threadIdx.x < blockDim.x - 1 ){ // if it is not the last thread
                x_next = s_mem[gene_block_id + 1];
                sum += 100*(xi*xi - x_next)*(xi*xi- x_next) + (xi - 1)*(xi - 1);
            }
        }
    }

    s_mem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, s_mem);

    if(threadIdx.x == 0){
        f[chromo_id] = s_mem[gene_block_id];
    }

}
#endif

#ifndef ELLIPS_KERNEL
#define ELLIPS_KERNEL
template <typename T>
__global__ void ellips_gpu(T *x, T *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x; 

    extern __shared__ T s_mem[];

    T xi = 0;
    T value = 0;
    
    for(i = threadIdx.x; i < nx; i += blockDim.x){
        xi = x[chromo_id*nx + i];

        value += xi*xi*pow(10.0, 6.0*i/(nx-1));
    }

    s_mem[gene_block_id] = value;
    __syncthreads();
    
    reduction(gene_block_id, s_mem);

    if(threadIdx.x == 0){
        f[chromo_id] = s_mem[gene_block_id];
    }    
}
#endif

#ifndef BENTCIGAR_KERNEL
#define BENTCIGAR_KERNEL
template <typename T>
__global__ void bent_cigar_gpu(T *x, T *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;
    
    extern __shared__ T s_mem[];

    T xi = 0.0;
    T x1 = 0.0;
    T sum = 0.0;

    if(threadIdx.x < nx){
        xi = x[chromo_id*nx + threadIdx.x];
        x1 = xi; // first thread will keep this value
        sum = xi*xi;
    }

    for(i = blockDim.x+threadIdx.x; i < nx; i+= blockDim.x){
        xi = x[chromo_id*nx + i];
        sum += xi*xi;  
    }

    s_mem[gene_block_id] = sum;
    __syncthreads();

    reduction(gene_block_id, s_mem);

    if(threadIdx.x == 0){
        f[chromo_id] = 1e6*s_mem[gene_block_id] - 1e6*x1*x1 + x1*x1;
    }

}
#endif

#ifndef DISCUS_KERNEL
#define DISCUS_KERNEL
template <typename T>
__global__ void discus_gpu(T *x, T *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;
    
    extern __shared__ T s_mem[];

    T xi = 0.0;
    T x1 = 0.0;
    T sum = 0.0;

    if(threadIdx.x < nx){
        xi = x[chromo_id*nx + threadIdx.x];
        x1 = xi; // first thread will keep this value
        sum = xi*xi;
    }

    for(i = blockDim.x+threadIdx.x; i < nx; i+= blockDim.x){
        xi = x[chromo_id*nx + i];
        sum += xi*xi;  
    }

    s_mem[gene_block_id] = sum;
    __syncthreads();

    reduction(gene_block_id, s_mem);

    if(threadIdx.x == 0){
        f[chromo_id] = s_mem[gene_block_id] - x1*x1 + 1e6*x1*x1;
    }

}
#endif