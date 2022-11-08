
#pragma once

#include "Benchmark.cuh"
#include <stdio.h>
#include "../benchmark_constants.cuh"
#include "../gpu_constants.cuh"
#include "../vector_ops.cuh"
#include "cublas_v2.h"


template <typename T>
__device__ T w_levy(T x);

template <typename T>
__global__ void levy_gpu(T *x, T *f, int nx);

template <class T> 
class Levy : public Benchmark<T> {
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
        
        Levy(int _n, int _pop_size){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);

            allocateMemory();

        }

        Levy(int _n, int _pop_size, char shift_filename[], char matrix_filename[]){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);
            
            cublasCreate(&(this->handle));

            this->use_rotation_matrix(matrix_filename, _n*_n);
            this->use_shift_vector(shift_filename, _n);            

            allocateMemory();

        }

        ~Levy(){
            freeMemory();
        }

        void compute(T *p_x_dev, T *p_f_dev){
            T* p_kernel_input;
            
            //shift
            if(this->shift_func){
                shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(p_x_dev, this->p_shift_dev, this->p_aux_dev, LEVY_BOUND/X_BOUND, this->n, this->pop_size);
            } else {
                //shrink
                shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(p_x_dev, this->p_aux_dev, LEVY_BOUND/X_BOUND, (this->n)*(this->pop_size));
            }

            if(this->rot_func){
                this->rotation(this->p_rotm_dev);
                p_kernel_input = this->rot_dev;
            } else {
                p_kernel_input = this->p_aux_dev;
            }
            
            levy_gpu<<<this->grid_size, this->block_shape, 2*(this->shared_mem_size)>>>(p_kernel_input, p_f_dev, this->n);
        }



};

template <typename T>
__global__ void levy_gpu(T *x, T *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;

    extern __shared__ T s_mem[];

    T wi   = 0.0;
    T sum = 0.0;

    if(threadIdx.x < nx){
        wi = w_levy(x[chromo_id*nx + threadIdx.x]);

        if(threadIdx.x == 0){
            sum = sin(PI*wi)*sin(PI*wi);
        }
        
        if(threadIdx.x == nx - 1){
            sum = (wi - 1)*(wi - 1)*(1 + sin(2*PI*wi)*sin(2*PI*wi));
        } else {
            sum += pow((wi-1),2) * (1+10*pow((sin(PI*wi+1)),2));
        }
    }


    for(i = threadIdx.x + blockDim.x; i < nx; i += blockDim.x){
        wi = w_levy(x[chromo_id*nx + i]);

        if(i == nx - 1){
            sum += (wi - 1)*(wi - 1)*(1 + sin(2*PI*wi)*sin(2*PI*wi));
        } else {
            sum += pow((wi-1),2) * (1+10*pow((sin(PI*wi+1)),2));
        }
    }


    s_mem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, s_mem);

    if(threadIdx.x == 0){
        f[chromo_id] = s_mem[gene_block_id];
    }

}

template <typename T>
__device__ T w_levy(T x){
    return 1 + (x - 0.0)/4.0;
}