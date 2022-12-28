
#pragma once

#include <stdio.h>
#include "Benchmark.cuh"

#ifndef ROSENBROCK_KERNEL
template <typename T>
__global__ void rosenbrock_gpu(T *x, T *f, int nx);
#endif

template <class T> 
class Rosenbrock : public Benchmark<T> {
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
        
        Rosenbrock(int _n, int _pop_size){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);

            allocateMemory();

        }

        Rosenbrock(int _n, int _pop_size, char shift_filename[], char matrix_filename[]){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);
            
            cublasCreate(&(this->handle));

            this->use_rotation_matrix(matrix_filename, _n*_n);
            this->use_shift_vector(shift_filename, _n);            

            allocateMemory();

        }


        ~Rosenbrock(){
            freeMemory();
        }

        void compute(T *p_x, T *p_f){
            T* p_kernel_input;

            this->checkPointers(p_x, p_f);

            //shift
            if(this->shift_func){
                shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, this->p_shift_dev, this->p_aux_dev, ROSENBROCK_BOUND/X_BOUND, this->n, this->pop_size);
            } else {
                //shrink
                shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, this->p_aux_dev, ROSENBROCK_BOUND/X_BOUND, (this->n)*(this->pop_size));
            }

            if(this->rot_func){
                this->rotation(this->p_rotm_dev);
                p_kernel_input = this->rot_dev;
            } else {
                p_kernel_input = this->p_aux_dev;
            }
            
            rosenbrock_gpu<<<this->grid_size, this->block_shape, 2*(this->shared_mem_size)>>>(p_kernel_input, this->p_f_dev, this->n);

            this->checkOutput(p_f);
        }



};

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
        xi = x[chromo_id*nx + threadIdx.x] + 1.0;
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
            s_mem[gene_block_id] = x[chromo_id*nx + i] + 1.0;
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