
#pragma once

#include <stdio.h>
#include "Benchmark.cuh"


#ifndef RASTRIGIN_KERNEL
template <typename T>
__global__ void step_shift_shrink_vector(T *x, T* shift_vector, T* out, float shrink_rate, int nx, int pop);

template <typename T>
__global__ void step_shrink_vector(T *x, T *out, float shrink_rate, int nx, int pop);

template <typename T>
__global__ void rastrigin_gpu(T *x, T *f, int nx);
#endif

template <class T> 
class StepRastrigin : public Benchmark<T> {
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

            this->freeIO();
            
        }
        
    public:
        
        StepRastrigin(int _n, int _pop_size){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);

            allocateMemory();

        }

        StepRastrigin(int _n, int _pop_size, char shift_filename[], char matrix_filename[]){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);
            
            cublasCreate(&(this->handle));

            this->use_rotation_matrix(matrix_filename, _n*_n);
            this->use_shift_vector(shift_filename, _n);            

            allocateMemory();

        }

        ~StepRastrigin(){
            freeMemory();
        }

        void compute(T *p_x, T *p_f){
            T* p_kernel_input;

            this->checkPointers(p_x, p_f);
            
            //shift
            if(this->shift_func){
                step_shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, this->p_shift_dev, this->p_aux_dev, RASTRIGIN_BOUND/X_BOUND, this->n, this->pop_size);
            } else {
                //shrink
                step_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, this->p_aux_dev, RASTRIGIN_BOUND/X_BOUND, (this->n), (this->pop_size));
            }

            if(this->rot_func){
                this->rotation(this->p_rotm_dev);
                p_kernel_input = this->rot_dev;
            } else {
                p_kernel_input = this->p_aux_dev;
            }
            
            rastrigin_gpu<<<this->grid_size, this->block_shape, 2*(this->shared_mem_size)>>>(p_kernel_input, this->p_f_dev, this->n);

            this->checkOutput(p_f);
        }



};

#ifndef RASTRIGIN_KERNEL
#define RASTRIGIN_KERNEL


template <typename T>
__global__ void step_shift_shrink_vector(T *x, T* shift_vector, T* out, float shrink_rate, int nx, int pop){
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    T step_x;
    T shift;

    if(tid < nx*pop){
        // shift vector and then shrink
        shift = shrink_rate*shift_vector[tid % nx];
        step_x = shrink_rate*x[tid];

		if (fabs(step_x-shift)>0.5){
            step_x = shift + floor(2*(step_x-shift)+0.5)/2;
        }
        
        out[tid] = (step_x - shift);
    }
}

template <typename T>
__global__ void step_shrink_vector(T *x, T *out, float shrink_rate, int nx, int pop){
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    T step_x;
    T shift;

    if(tid < nx*pop){
        // shrink
        shift = 0.0;
        step_x = shrink_rate*x[tid];

		if (fabs(step_x-shift)>0.5){
            step_x = shift + floor(2*(step_x-shift)+0.5)/2;
        }
        
        out[tid] = (step_x - shift);
    }
}

template <typename T>
__global__ void rastrigin_gpu(T *x, T *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;

    extern __shared__ T s_mem[];

    T xi = 0;
    T value = 0;
    
    for(i = threadIdx.x; i < nx; i += blockDim.x){
        xi = x[chromo_id*nx + i];

        value += xi*xi - 10*cos(2*PI*xi) + 10;
    }

    s_mem[gene_block_id] = value;
    __syncthreads();
    
    reduction(gene_block_id, s_mem);

    if(threadIdx.x == 0){
        f[chromo_id] = s_mem[gene_block_id];
    }    
}
#endif