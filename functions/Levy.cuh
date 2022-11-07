
#pragma once

#include "Benchmark.cuh"
#include <stdio.h>
#include "../benchmark_constants.cuh"
#include "../gpu_constants.cuh"
#include "../benchmark_kernels.cuh"
#include "../vector_ops.cuh"
#include "cublas_v2.h"

template <class T> 
class Levy : public Benchmark<T> {
    private:
        void allocateMemory(){
            cudaMalloc<T>(&(this->p_x_dev), (this->n)*(this->pop_size)*sizeof(T));
            cudaMalloc<T>(&(this->p_aux_dev), (this->n)*(this->pop_size)*sizeof(T));
            cudaMalloc<T>(&(this->p_f_dev), (this->pop_size)*sizeof(T));
        }

        void freeMemory(){
            cudaFree(this->p_x_dev);
            cudaFree(this->p_aux_dev);
            cudaFree(this->p_f_dev);

            cublasDestroy(this->handle);
            
            if(this->shift_func) cudaFree(this->p_shift_dev);
            if(this->rot_func)   cudaFree(this->p_rotm_dev);
        }
    public:
        
        Levy(int _n, int _pop_size){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);
            
            cublasCreate(&(this->handle));

            char matrix_filename[50] = {};

            snprintf(matrix_filename, 50, "./input_data/matrices/basic_%d.bin", _n);
            this->use_rotation_matrix(matrix_filename, _n*_n);

            this->use_shift_vector("./input_data/shift_vectors/basic_shift.bin", _n);            

            allocateMemory();

        }

        ~Levy(){
            freeMemory();
        }

        void compute(){
            T* p_kernel_input;
            
            //shift
            if(this->shift_func){
                shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, this->p_shift_dev, this->p_aux_dev, LEVY_BOUND/X_BOUND, this->n, this->pop_size);
            } else {
                //shrink
                shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, this->p_aux_dev, LEVY_BOUND/X_BOUND, (this->n)*(this->pop_size));
            }

            if(this->rot_func){
                this->rotation(this->p_rotm_dev);
                p_kernel_input = this->rot_dev;
            } else {
                p_kernel_input = this->p_aux_dev;
            }
            
            levy_gpu<<<this->grid_size, this->block_shape, 2*(this->shared_mem_size)>>>(p_kernel_input, this->p_f_dev, this->n);
        }



};
