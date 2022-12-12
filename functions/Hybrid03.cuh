
#pragma once

#include "Benchmark.cuh"
#include <stdio.h>
#include "../benchmark_constants.cuh"
#include "../gpu_constants.cuh"
#include "../vector_ops.cuh"
#include "cublas_v2.h"


template <typename T>
__global__ void hf06_gpu(T *x, T *f, int nx);

template <class T> 
class Hybrid03 : public Benchmark<T> {
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
        
        Hybrid03(int _n, int _pop_size){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);

            allocateMemory();

        }

        Hybrid03(int _n, int _pop_size, char shuffle_filename[], char shift_filename[], char matrix_filename[]){
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

        ~Hybrid03(){
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
            
            shuffle_vector<<<this->pop_size, MIN_OCCUPANCY>>>(p_kernel_input, this->p_shuffle_dev, p_kernel_input, this->n, this->pop_size);

            hf06_gpu<<<this->grid_size, this->block_shape, 2*(this->shared_mem_size)>>>(p_kernel_input, this->p_f_dev, this->n);

            this->checkOutput(p_f);
        }



};

template <typename T>
__global__ void hf06_gpu(T *x, T *f, int nx){
    #define HF06_n1 0.3
    #define HF06_n2 0.2
    #define HF06_n3 0.2
    #define HF06_n4 0.1

    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;

    extern __shared__ T smem[];

    T xi;
    T sum = 0.0;
    T sum2 = 0.0;
    T fit;

    x = &x[chromo_id*nx];

    // KATSUURA
    int i = threadIdx.x;
    int n_f = ceil(nx*HF06_n1);
    sum = 1.0; // a multiplication is computed
    for( ; i < n_f; i += blockDim.x){
        T sumJ = 0.0;

        xi = x[i]*KATSUURA_BOUND/X_BOUND;

        for(int j = 1; j <= 32; j ++){
            sumJ += fabs(pow(2.0,j)*xi - round(pow(2.0,j)*xi))/pow(2.0,j); 
        }

        sum *= pow( 1.0+(i+1)*sumJ , 10.0/pow(n_f, 1.2) );
    }


    smem[gene_block_id] = sum;
    __syncthreads();
    reduction_mult(gene_block_id, smem);

    if(threadIdx.x == 0){
        fit = (10.0/(n_f*n_f))*smem[gene_block_id] - 10.0/(n_f*n_f);

    }

    // HAPPYCAT
    i = threadIdx.x + n_f;
    n_f += ceil(nx*HF06_n2);
    sum = 0.0;
    sum2 = 0.0;    
    for( ; i < n_f; i += blockDim.x){

        xi = x[i]*HAPPYCAT_BOUND/X_BOUND - 1.0;

        sum  += xi*xi;
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

        fit += pow(fabs(sum - ceil(nx*HF06_n2)), 0.25) + (0.5*sum + sum2)/ceil(nx*HF06_n2) + 0.5;

    }

    // GRIEROSEN
    i = threadIdx.x + n_f;
    T x1 = x[n_f]*GRIE_ROSEN_BOUND/X_BOUND + 1;
   
    n_f += ceil(nx*HF06_n3);
   
    T x_next = 0.0;
    sum = 0.0;
    for(; i < n_f; i += blockDim.x){
        xi = x[i]*GRIE_ROSEN_BOUND/X_BOUND + 1;
        // if(i == n_f-1){
        //     x1 = x[i - ((int)ceil(nx*HF06_n3)-1)]*GRIE_ROSEN_BOUND/X_BOUND + 1;    
        // }

        if(i == n_f-1){
            x_next = x1;
        } else {
            x_next = x[i+1]*GRIE_ROSEN_BOUND/X_BOUND  +1;
        }

        sum2 = 100*pow((xi*xi - x_next), 2) + (xi-1)*(xi-1); // rosenbrock(x1, x2)
        sum2 = sum2*sum2/4000 - cos(sum2) + 1; // griewank(x1)
        sum += sum2;
    } 
    smem[gene_block_id] = sum;

    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        fit += smem[gene_block_id]; 
    }

    // SCHWEFEL
    i = threadIdx.x + n_f;
    n_f += ceil(nx*HF06_n4);
    sum = 0.0;
    T zi;
    for( ; i < n_f; i += blockDim.x){
        zi = 4.209687462275036e+002 + x[i]*SCHWEFEL_BOUND/X_BOUND;
        T zi_fmod = fmod(zi, 500.0);
        if(fabs(zi) <= 500.0){
            sum += zi*sin(pow(fabs(zi),0.5));
        } else if(zi > 500.0) {
            sum += (500 - zi_fmod)*sin(pow(fabs(500 - zi_fmod), 0.5)) - (zi - 500)*(zi - 500)/(10000*ceil(nx*HF06_n4));
        } else{
            sum += (-500.0+fmod(fabs(zi),500.0))*sin(pow(500.0-fmod(fabs(zi),500.0),0.5)) - (zi + 500)*(zi + 500)/(10000*ceil(nx*HF06_n4));
        }
    }

    smem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        fit += 4.189828872724338e+002 * ceil(nx*HF06_n4) - smem[gene_block_id];

    }

    // ACKLEY
    i = threadIdx.x + n_f;
    sum = 0.0;
    sum2 = 0.0;
    for( ; i < nx; i += blockDim.x){
        xi = x[i]*ACKLEY_BOUND/X_BOUND;

        sum += xi*xi;
        sum2 += cos(2*PI*xi);
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
        T c = 1.0 / float(nx-n_f);

        f[chromo_id]  = fit + 20 - 20*exp(-0.2*sqrt( c * sum)) + E - exp(c * sum2);  

    }
}
