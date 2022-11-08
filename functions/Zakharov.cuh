
#pragma once

#include "Benchmark.cuh"
#include <stdio.h>
#include "../benchmark_constants.cuh"
#include "../gpu_constants.cuh"
#include "../vector_ops.cuh"
#include "cublas_v2.h"

template <typename T>
__global__ void zakharov_gpu(T *x, T *f, int nx);

template <class T> 
class Zakharov : public Benchmark<T> {
    private:
        void allocateMemory(){
            cudaMalloc<T>(&(this->p_aux_dev), (this->n)*(this->pop_size)*sizeof(T));
        }

        void freeMemory(){
            cudaFree(this->p_aux_dev);
            
            if(this->rot_func){
                cudaFree(this->p_rotm_dev);
                cublasDestroy(this->handle);
            }
            
            if(this->shift_func) cudaFree(this->p_shift_dev);
            
        }
        
    public:
        
        Zakharov(int _n, int _pop_size){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);

            allocateMemory();

        }

        Zakharov(int _n, int _pop_size, char shift_filename[], char matrix_filename[]){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);
            
            cublasCreate(&(this->handle));

            this->use_rotation_matrix(matrix_filename, _n*_n);
            this->use_shift_vector(shift_filename, _n);            

            allocateMemory();

        }

        ~Zakharov(){
            freeMemory();
        }

        void compute(T *p_x_dev, T *p_f_dev){
            T* p_kernel_input;
            
            //shift
            if(this->shift_func){
                shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(p_x_dev, this->p_shift_dev, this->p_aux_dev, ZAKHAROV_BOUND/X_BOUND, this->n, this->pop_size);
            } else {
                //shrink
                shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(p_x_dev, this->p_aux_dev, ZAKHAROV_BOUND/X_BOUND, (this->n)*(this->pop_size));
            }

            if(this->rot_func){
                this->rotation(this->p_rotm_dev);
                p_kernel_input = this->rot_dev;
            } else {
                p_kernel_input = this->p_aux_dev;
            }
            
            zakharov_gpu<<<this->grid_size, this->block_shape, 2*(this->shared_mem_size)>>>(p_kernel_input, p_f_dev, this->n);
        }



};


template <> 
__global__ void zakharov_gpu<double>(double *x, double *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;

    extern __shared__ double2 smemvec[];

    double2 sum = {0, 0};

    double value = 0;
    
    for(i = threadIdx.x; i < nx; i += blockDim.x){
        value = x[chromo_id*nx + i];

        sum.x += value*value;
        sum.y += 0.5*(i+1)*value;
    }

    smemvec[gene_block_id] = sum;
    __syncthreads();
    
    for( i = blockDim.x / 2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            smemvec[gene_block_id].x += smemvec[gene_block_id + i].x;
            smemvec[gene_block_id].y += smemvec[gene_block_id + i].y;
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        sum = smemvec[gene_block_id];
        sum.y = sum.y*sum.y;
        f[chromo_id] = sum.x + sum.y + sum.y*sum.y; 
    }

}

template <> 
__global__ void zakharov_gpu<float>(float *x, float *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;

    extern __shared__ float2 smemvec[];

    float2 sum = {0, 0};

    float value = 0;
    
    for(i = threadIdx.x; i < nx; i += blockDim.x){
        value = x[chromo_id*nx + i];

        sum.x += value*value;
        sum.y += 0.5*(i+1)*value;
    }

    smemvec[gene_block_id] = sum;
    __syncthreads();
    
    for( i = blockDim.x / 2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            smemvec[gene_block_id].x += smemvec[gene_block_id + i].x;
            smemvec[gene_block_id].y += smemvec[gene_block_id + i].y;
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        sum = smemvec[gene_block_id];
        sum.y = sum.y*sum.y;
        f[chromo_id] = sum.x + sum.y + sum.y*sum.y; 
    }

}
