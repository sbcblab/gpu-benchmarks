
#pragma once

#include "Benchmark.cuh"
#include <stdio.h>
#include "../benchmark_constants.cuh"
#include "../gpu_constants.cuh"
#include "../benchmark_kernels.cuh"
#include "../vector_ops.cuh"
#include "cublas_v2.h"


template <class T> 
class Composition02 : public Benchmark<T> {
    private:
        int cf_num = 3; // number of functions that are part of the composition function
        T * p_cfit_dev;
        T * p_tcfit_dev;

        void allocateMemory(){
            cudaMalloc<T>(&(this->p_x_dev), (this->n)*(this->pop_size)*sizeof(T));
            cudaMalloc<T>(&(this->p_aux_dev), (this->n)*(this->pop_size)*sizeof(T));
            cudaMalloc<T>(&(this->p_f_dev), (this->pop_size)*sizeof(T));
            cudaMalloc<T>(&p_tcfit_dev, cf_num*(this->pop_size)*sizeof(T));
            cudaMalloc<T>(&p_cfit_dev, cf_num*(this->pop_size)*sizeof(T));
        }

        void freeMemory(){
            cudaFree(this->p_x_dev);
            cudaFree(this->p_aux_dev);
            cudaFree(this->p_f_dev);
            cudaFree(p_cfit_dev);
            cudaFree(p_tcfit_dev);

            cublasDestroy(this->handle);
            
            if(this->shift_func) cudaFree(this->p_shift_dev);
            if(this->rot_func)   cudaFree(this->p_rotm_dev);
        }


        void transpose_fit(){
            
        }

    public:
        
        Composition02(int _n, int _pop_size){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);
            
            cublasCreate(&(this->handle));

            char matrix_filename[50] = {};

            snprintf(matrix_filename, 50, "./input_data/matrices/composition_%d.bin", _n);
            this->use_rotation_matrix(matrix_filename, _n*_n*cf_num);

            this->use_shift_vector("./input_data/shift_vectors/composition_shift.bin", _n*cf_num);            

            allocateMemory();


            float delta[3] = {20,10,10};
            float bias[3]  = {0, 200, 100};
            float lambda[3] = {1, 1, 1};

            // memcpys may ovewrite useful data in constant memory if multiple composition functions are instantiated
            cudaMemcpyToSymbol(p_delta_dev, delta, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(p_bias_dev, bias, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(p_lambda_dev, lambda, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);

        }

        ~Composition02(){
            freeMemory();
        }


        void compute(){
            
            int offset = 0;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, this->p_shift_dev, this->p_aux_dev, SCHWEFEL_BOUND/X_BOUND, this->n, this->pop_size);
            schwefel_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->p_aux_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            
            offset = 1;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, &(this->p_shift_dev[offset*(this->n)]), this->p_aux_dev, RASTRIGIN_BOUND/X_BOUND, this->n, this->pop_size);
            this->rotation(&(this->p_rotm_dev[offset*this->n*this->n]));
            rastrigin_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->rot_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            
            offset = 2;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, &(this->p_shift_dev[offset*(this->n)]), this->p_aux_dev, HGBAT_BOUND/X_BOUND, this->n, this->pop_size);
            this->rotation(&(this->p_rotm_dev[offset*this->n*this->n]));
            hgbat_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->rot_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            
            transpose_fit();
            dim3 BLOCK(32, cf_num);

            cfcal_gpu<<<this->pop_size, BLOCK>>>(  this->p_x_dev, 
                                                    this->p_f_dev, 
                                                    this->p_shift_dev, 
                                                    this->p_tcfit_dev, 
                                                    this->n );
        }

};


template<> 
void Composition02<double>::transpose_fit(){
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
void Composition02<float>::transpose_fit(){
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