
#pragma once

#include <stdio.h>
#include "Benchmark.cuh"


#ifndef SCHAFFERF7_KERNEL
template <typename T>
__global__ void schaffer_F7_gpu(T *x, T *f, int nx, int constant_f);
#endif

template <class T> 
class SchafferF7 : public Benchmark<T> {
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
        
        SchafferF7(int _n, int _pop_size){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);

            allocateMemory();

        }

        SchafferF7(int _n, int _pop_size, char shift_filename[], char matrix_filename[]){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);
            
            cublasCreate(&(this->handle));

            this->use_rotation_matrix(matrix_filename, _n*_n);
            this->use_shift_vector(shift_filename, _n);            

            allocateMemory();

        }

        ~SchafferF7(){
            freeMemory();
        }

        void compute(T *p_x, T *p_f){
            T* p_kernel_input;

            this->checkPointers(p_x, p_f);
            
            //shift
            if(this->shift_func){
                shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, this->p_shift_dev, this->p_aux_dev, ESCAFFER6_BOUND/X_BOUND, this->n, this->pop_size);
            } else {
                //shrink
                shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, this->p_aux_dev, ESCAFFER6_BOUND/X_BOUND, (this->n)*(this->pop_size));
            }

            if(this->rot_func){
                this->rotation(this->p_rotm_dev);
                p_kernel_input = this->rot_dev;
            } else {
                p_kernel_input = this->p_aux_dev;
            }
            
            schaffer_F7_gpu<<<this->grid_size, this->block_shape, 2*(this->shared_mem_size)>>>(p_kernel_input, this->p_f_dev, this->n, C_SCHAFFER_F7);
            

            this->checkOutput(p_f);
        }



};

#ifndef SCHAFFERF7_KERNEL
#define SCHAFFERF7_KERNEL
template <typename T>
__global__ void schaffer_F7_gpu(T *x, T *f, int nx, int constant_f){
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;

    extern __shared__ T s_mem[];

    T sum = 0.0;

    s_mem[gene_block_id] = 0.0;
    __syncthreads(); 

        
    for(int i = threadIdx.x; i < nx - 1; i += blockDim.x){
        // talvez a cache ajude nessa operação
        T si = pow(x[chromo_id*nx + i]*x[chromo_id*nx + i] + x[chromo_id*nx + i +1]*x[chromo_id*nx + i + 1], 0.5);
        sum += pow(si, 0.5) + pow(si, 0.5)*sin(50.0*pow(si, 0.2))*sin(50.0*pow(si, 0.2));
        
    }

    s_mem[gene_block_id] = sum;
    __syncthreads();

    reduction(gene_block_id, s_mem);

    if(threadIdx.x == 0){

        f[chromo_id] = s_mem[gene_block_id]*s_mem[gene_block_id]/((nx-1)*(nx-1)) + constant_f;
    }
    
}
#endif