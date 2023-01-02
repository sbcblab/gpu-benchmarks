
#pragma once

#include <stdio.h>

#include "Benchmark.cuh"


#ifndef ROSENBROCK_KERNEL_NO_CONSTANT
template <typename T>
__global__ void rosenbrock_gpu(T *x, T *f, int nx);
#endif

#ifndef SCHWEFEL_KERNEL_NO_CONSTANT
template <typename T>
__global__ void schwefel_gpu(T *x, T *f, int nx);
#endif

#ifndef RASTRIGIN_KERNEL_NO_CONSTANT
template <typename T>
__global__ void rastrigin_gpu(T *x, T *f, int nx);
#endif

#ifndef GRIEWANK_KERNEL_NO_CONSTANT
template <typename T>
__global__ void griewank_gpu(T *x, T *f, int nx);
#endif

#ifndef ESCAFFER6_KERNEL_NO_CONSTANT
template <typename T>
__global__ void escaffer6_gpu(T *x, T *f, int nx);

template <typename T>
__device__ T g_schaffer_f6(T x, T y);
#endif

template <class T> 
class Composition03 : public Benchmark<T> {
    private:
        int cf_num = 5; // number of functions that are part of the composition function
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


        void transpose_fit(){
            
        }

    public:
        
        Composition03(int _n, int _pop_size){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);
            
            cublasCreate(&(this->handle));

            allocateMemory();

            float delta[5]  = {20,20,30,30,20};
            float bias[5]   = {0, 200, 300, 400, 200};
            float lambda[5] = {10000/2e+7, 1, 1000/100, 1, 10000/1e+3};

            // memcpys may ovewrite useful data in constant memory if multiple composition functions are instantiated
            cudaMemcpyToSymbol(p_delta_dev, delta, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(p_bias_dev, bias, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(p_lambda_dev, lambda, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);

        }

        Composition03(int _n, int _pop_size, char shift_filename[], char matrix_filename[]){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);
            
            cublasCreate(&(this->handle));

            this->use_rotation_matrix(matrix_filename, _n*_n*cf_num);
            this->use_shift_vector(shift_filename, _n*cf_num);            

            allocateMemory();

            float delta[5]  = {20,20,30,30,20};
            float bias[5]   = {0, 200, 300, 400, 200};
            float lambda[5] = {10000/2e+7, 1, 1000/100, 1, 10000/1e+3};

            // memcpys may ovewrite useful data in constant memory if multiple composition functions are instantiated
            cudaMemcpyToSymbol(p_delta_dev, delta, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(p_bias_dev, bias, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(p_lambda_dev, lambda, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);

        }

        ~Composition03(){
            freeMemory();
        }


        void compute(T *p_x, T *p_f){

            this->checkPointers(p_x, p_f);

            int offset = 0;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, this->p_shift_dev, this->p_aux_dev, ESCAFFER6_BOUND/X_BOUND, this->n, this->pop_size);
            this->rotation(this->p_rotm_dev);
            escaffer6_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->rot_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            
            offset = 1;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, &(this->p_shift_dev[offset*(this->n)]), this->p_aux_dev, SCHWEFEL_BOUND/X_BOUND, this->n, this->pop_size);
            this->rotation(&(this->p_rotm_dev[offset*this->n*this->n]));
            schwefel_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->rot_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            
            offset = 2;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, &(this->p_shift_dev[offset*(this->n)]), this->p_aux_dev, GRIEWANK_BOUND/X_BOUND, this->n, this->pop_size);
            this->rotation(&(this->p_rotm_dev[offset*this->n*this->n]));
            griewank_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->rot_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            
            offset = 3;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, &(this->p_shift_dev[offset*(this->n)]), this->p_aux_dev, ROSENBROCK_BOUND/X_BOUND, this->n, this->pop_size);
            this->rotation(&(this->p_rotm_dev[offset*this->n*this->n]));
            rosenbrock_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->rot_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            
            offset = 4;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, &(this->p_shift_dev[offset*(this->n)]), this->p_aux_dev, RASTRIGIN_BOUND/X_BOUND, this->n, this->pop_size);
            this->rotation(&(this->p_rotm_dev[offset*this->n*this->n]));
            rastrigin_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->rot_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            
            transpose_fit();
            dim3 BLOCK(32, cf_num);

            cfcal_gpu<<<this->pop_size, BLOCK>>>(  this->p_x_dev, 
                                                    this->p_f_dev, 
                                                    this->p_shift_dev, 
                                                    this->p_tcfit_dev, 
                                                    this->n,
                                                    C_COMPOSITION3 );

            this->checkOutput(p_f);
        }

};


template<> 
void Composition03<double>::transpose_fit(){
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
void Composition03<float>::transpose_fit(){
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

#ifndef GRIEWANK_KERNEL_NO_CONSTANT
#define GRIEWANK_KERNEL_NO_CONSTANT
template <typename T>
__global__ void griewank_gpu(T *x, T *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x; 

    extern __shared__ T s_mem[];

    T xi;

    T term1 = 0.0;
    T term2 = 1.0;

    for(i = threadIdx.x; i < nx; i += blockDim.x){
        xi = x[chromo_id*nx + i];

        term1 += xi*xi;

        term2 *= cos(xi/pow(i+1.0, 0.5));
    }


    s_mem[gene_block_id] = term1;
    __syncthreads();
    reduction(gene_block_id, s_mem);
    
    if(threadIdx.x == 0){
        term1 = s_mem[gene_block_id];
    }

    s_mem[gene_block_id] = term2;
    __syncthreads();
    reduction_mult(gene_block_id, s_mem);
    
    if(threadIdx.x == 0){
        term2 = s_mem[gene_block_id];
        f[chromo_id] = term1/4000 - term2 + 1;
    }
}
#endif

#ifndef ROSENBROCK_KERNEL_NO_CONSTANT
#define ROSENBROCK_KERNEL_NO_CONSTANT
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

#ifndef SCHWEFEL_KERNEL_NO_CONSTANT
#define SCHWEFEL_KERNEL_NO_CONSTANT
template <typename T>
__global__ void schwefel_gpu(T *x, T *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;
    
    extern __shared__ T s_mem[];

    T sum = 0.0;
    T zi;
    for(i = threadIdx.x ; i < nx; i += blockDim.x){
        zi = 4.209687462275036e+002 + x[chromo_id*nx + i];
        T zi_fmod = fmod(zi, 500.0);
        if(fabs(zi) <= 500.0){
            sum += zi*sin(pow(fabs(zi),0.5));
        } else if(zi > 500.0) {
            sum += (500 - zi_fmod)*sin(pow(fabs(500 - zi_fmod), 0.5)) - (zi - 500)*(zi - 500)/(10000*nx);
        } else{
            sum += (-500.0+fmod(fabs(zi),500.0))*sin(pow(500.0-fmod(fabs(zi),500.0),0.5)) - (zi + 500)*(zi + 500)/(10000*nx);
        }
    }

    s_mem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, s_mem);

    if(threadIdx.x == 0){
        f[chromo_id] = 4.189828872724338e+002 * nx - s_mem[gene_block_id];
    }
}
#endif

#ifndef RASTRIGIN_KERNEL_NO_CONSTANT
#define RASTRIGIN_KERNEL_NO_CONSTANT
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

#ifndef ESCAFFER6_KERNEL_NO_CONSTANT
#define ESCAFFER6_KERNEL_NO_CONSTANT
template <typename T>
__global__ void escaffer6_gpu(T *x, T *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;
    
    extern __shared__ T s_mem[];

    T xi   = 0;
    T xi_1 = 0;
    T sum  = 0;
    const int n_blockdims = (int)(blockDim.x*ceil((float)nx/blockDim.x));

    if(threadIdx.x < nx){
        xi = x[chromo_id*nx + (threadIdx.x % nx)];
        xi_1 = x[chromo_id*nx + (threadIdx.x+1)%nx];

        sum = g_schaffer_f6(xi, xi_1);
    }

    // every thread in a warp enters in this for 
    for(i = blockDim.x + threadIdx.x; i < n_blockdims; i+= blockDim.x){
        
        if(i < nx){
            xi = x[chromo_id*nx + (i % nx)];
            xi_1 = x[chromo_id*nx + (i+1)%nx];

            sum += g_schaffer_f6(xi, xi_1);
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
__device__ T g_schaffer_f6(T x, T y){
    T num = sin(sqrt(x*x + y*y))*sin(sqrt(x*x + y*y)) - 0.5;
    T dem = (1 + 0.001*(x*x + y*y))*(1 + 0.001*(x*x + y*y));
    return 0.5 + num/dem;
}
#endif
