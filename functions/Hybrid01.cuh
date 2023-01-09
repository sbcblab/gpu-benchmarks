
#pragma once

#include <stdio.h>

#include "Benchmark.cuh"

template <typename T>
__global__ void hf02_gpu(T *x, T *f, int nx);

template <class T> 
class Hybrid01 : public Benchmark<T> {
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
        
        Hybrid01(int _n, int _pop_size){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);

            allocateMemory();

        }

        Hybrid01(int _n, int _pop_size, char shuffle_filename[], char shift_filename[], char matrix_filename[]){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);
            
            cublasCreate(&(this->handle));

            this->use_rotation_matrix(matrix_filename, _n*_n);
            this->use_shift_vector(shift_filename, _n);
            this->use_shuffle_vector(shuffle_filename, _n);       

            allocateMemory();

        }

        ~Hybrid01(){
            freeMemory();
        }

        void compute(T *p_x, T *p_f){
            T* p_kernel_input;
            
            this->checkPointers(p_x, p_f);

            //shift
            if(this->shift_func){
                shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, this->p_shift_dev, this->p_aux_dev, 1, this->n, this->pop_size);
            } 

            if(this->rot_func){
                this->rotation(this->p_rotm_dev);
                p_kernel_input = this->rot_dev;
            } else {
                p_kernel_input = this->p_aux_dev;
            }

            shuffle_vector<<<(this->pop_size), MIN_OCCUPANCY>>>(p_kernel_input, this->p_shuffle_dev, p_kernel_input, this->n, this->pop_size);
            
            hf02_gpu<<<this->grid_size, this->block_shape, 2*(this->shared_mem_size)>>>(p_kernel_input, this->p_f_dev, this->n);

            this->checkOutput(p_f);
        }

};

template <typename T>
__global__ void hf02_gpu(T *x, T *f, int nx){
    #define HF02_n1 0.4
    #define HF02_n2 0.4
    #define HF02_n3 0.2
    
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;

    extern __shared__ T smem[];

    T xi, x1;
    T sum  = 0.0;
    T sum2 = 0.0;
    T fit;

    int i = threadIdx.x;
    int n_f = ceil(nx*HF02_n1);

    x = &x[chromo_id*nx];

    // bent-cigar

    if( i < n_f ){
        xi = x[i];
        x1 = xi;
        sum = xi*xi;
    }

    i += blockDim.x;
    for(; i < n_f; i += blockDim.x){
        xi = x[i];
        sum += xi*xi;
    }

    smem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, smem);
    
    if(threadIdx.x == 0){
        fit = 1e6*smem[gene_block_id] - 1e6*x1*x1 + x1*x1;
    }


    // hg-bat
    i = threadIdx.x + n_f;
    n_f += ceil(nx*HF02_n2);
    sum = 0.0;
    sum2 = 0.0;
    for(; i < n_f; i += blockDim.x){
        xi = x[i]*HGBAT_BOUND/X_BOUND - 1.0;

        sum += xi*xi;
        sum2 += xi;
    }

    smem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        sum = smem[gene_block_id];
    }

    smem[gene_block_id] = sum2;
    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        sum2 = smem[gene_block_id];
        fit += sqrt(fabs(sum*sum - sum2*sum2)) + (0.5*sum + sum2)/ceil(nx*HF02_n2); + 0.5;
    }

  
    // rastrigin
    i = threadIdx.x + n_f;
    sum = 0.0;
    for(; i < nx; i += blockDim.x){
        xi = x[i]*RASTRIGIN_BOUND/X_BOUND;

        sum += xi*xi - 10*cos(2*PI*xi) + 10;
    }

    smem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        f[chromo_id] = smem[gene_block_id] + fit + C_HYBRID1;
    }
}

