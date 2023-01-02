
#pragma once

#include <stdio.h>
#include "Benchmark.cuh"

#ifndef HGBAT_KERNEL_NO_CONSTANT
template<typename T>
__global__ void hgbat_gpu(T *x, T *f, int nx);
#endif

#ifndef SCHWEFEL_KERNEL_NO_CONSTANT
template <typename T>
__global__ void schwefel_gpu(T *x, T *f, int nx);
#endif

#ifndef RASTRIGIN_KERNEL_NO_CONSTANT
template <typename T>
__global__ void rastrigin_gpu(T *x, T *f, int nx);
#endif

#ifndef BENTCIGAR_KERNEL_NO_CONSTANT
template <typename T>
__global__ void bent_cigar_gpu(T *x, T *f, int nx);
#endif

#ifndef ELLIPS_KERNEL_NO_CONSTANT
template <typename T>
__global__ void ellips_gpu(T *x, T *f, int nx);
#endif

#ifndef ESCAFFER6_KERNEL_NO_CONSTANT
template <typename T>
__global__ void escaffer6_gpu(T *x, T *f, int nx);

template <typename T>
__device__ T g_schaffer_f6(T x, T y);
#endif

template <class T> 
class Composition04 : public Benchmark<T> {
    private:
        int cf_num = 6; // number of functions that are part of the composition function
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
        
        Composition04(int _n, int _pop_size){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);
            
            cublasCreate(&(this->handle));

            allocateMemory();

            float delta[6] = {10, 20, 30, 40, 50, 60};
            float bias[6]  = {0, 300, 500, 100, 400, 200};
            float lambda[6] = {10000/1000, 10000/1e+3, 10000/4e+3, 10000/1e+30, 10000/1e+10, 10000/2e+7};

            // memcpys may ovewrite useful data in constant memory if multiple composition functions are instantiated
            cudaMemcpyToSymbol(p_delta_dev, delta, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(p_bias_dev, bias, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(p_lambda_dev, lambda, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);

        }

        Composition04(int _n, int _pop_size, char shift_filename[], char matrix_filename[]){
            this->pop_size = _pop_size;
            this->n = _n;

            this->grid_size_shift = (_pop_size*_n)/MIN_OCCUPANCY + int((_pop_size*_n % MIN_OCCUPANCY) > 0);

            this->kernel_launch_config(this->grid_size, this->block_shape, this->shared_mem_size);
            
            cublasCreate(&(this->handle));

            this->use_rotation_matrix(matrix_filename, _n*_n*cf_num);
            this->use_shift_vector(shift_filename, _n*cf_num);     

            allocateMemory();

            float delta[6] = {10, 20, 30, 40, 50, 60};
            float bias[6]  = {0, 300, 500, 100, 400, 200};
            float lambda[6] = {10000/1000, 10000/1e+3, 10000/4e+3, 10000/1e+30, 10000/1e+10, 10000/2e+7};

            // memcpys may ovewrite useful data in constant memory if multiple composition functions are instantiated
            cudaMemcpyToSymbol(p_delta_dev, delta, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(p_bias_dev, bias, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(p_lambda_dev, lambda, cf_num*sizeof(float), 0, cudaMemcpyHostToDevice);

        }

        ~Composition04(){
            freeMemory();
        }


        void compute(T *p_x, T *p_f){

            this->checkPointers(p_x, p_f);
            
            int offset = 0;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, this->p_shift_dev, this->p_aux_dev, HGBAT_BOUND/X_BOUND, this->n, this->pop_size);
            this->rotation(this->p_rotm_dev);
            hgbat_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->rot_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            
            offset = 1;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, &(this->p_shift_dev[offset*(this->n)]), this->p_aux_dev, RASTRIGIN_BOUND/X_BOUND, this->n, this->pop_size);
            this->rotation(&(this->p_rotm_dev[offset*this->n*this->n]));
            rastrigin_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->rot_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            
            offset = 2;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, &(this->p_shift_dev[offset*(this->n)]), this->p_aux_dev, SCHWEFEL_BOUND/X_BOUND, this->n, this->pop_size);
            this->rotation(&(this->p_rotm_dev[offset*this->n*this->n]));
            schwefel_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->rot_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            
            offset = 3;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, &(this->p_shift_dev[offset*(this->n)]), this->p_aux_dev, BENT_CIGAR_BOUND/X_BOUND, this->n, this->pop_size);
            this->rotation(&(this->p_rotm_dev[offset*this->n*this->n]));
            bent_cigar_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->rot_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            
            offset = 4;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, &(this->p_shift_dev[offset*(this->n)]), this->p_aux_dev, ELLIPSIS_BOUND/X_BOUND, this->n, this->pop_size);
            this->rotation(&(this->p_rotm_dev[offset*this->n*this->n]));
            ellips_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->rot_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            
            offset = 5;
            shift_shrink_vector<<<this->grid_size_shift, MIN_OCCUPANCY>>>(this->p_x_dev, &(this->p_shift_dev[offset*(this->n)]), this->p_aux_dev, ESCAFFER6_BOUND/X_BOUND, this->n, this->pop_size);
            this->rotation(&(this->p_rotm_dev[offset*this->n*this->n]));
            escaffer6_gpu<<<this->grid_size, this->block_shape, 2*this->shared_mem_size>>>(this->rot_dev, &(this->p_cfit_dev[offset*this->pop_size]), this->n);
            

            transpose_fit();
            dim3 BLOCK(32, cf_num);

            cfcal_gpu<<<this->pop_size, BLOCK>>>(  this->p_x_dev, 
                                                    this->p_f_dev, 
                                                    this->p_shift_dev, 
                                                    this->p_tcfit_dev, 
                                                    this->n,
                                                    C_COMPOSITION4 );

            this->checkOutput(p_f);
        }

};


template<> 
void Composition04<double>::transpose_fit(){
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
void Composition04<float>::transpose_fit(){
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

#ifndef HGBAT_KERNEL_NO_CONSTANT
#define HGBAT_KERNEL_NO_CONSTANT
template<>
__global__ void hgbat_gpu<double>(double *x, double *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;

    extern __shared__ double2 smemvec_d[];

    double xi   = 0;
    double2 sum  = {0, 0};

    for(i = threadIdx.x; i < nx; i += blockDim.x){
        xi = x[chromo_id*nx + i] - 1.0;

        sum.x  += xi*xi;
        sum.y  += xi;
    }

    smemvec_d[gene_block_id] = sum;
    __syncthreads();
    
    for( i = blockDim.x / 2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            smemvec_d[gene_block_id].x += smemvec_d[gene_block_id + i].x;
            smemvec_d[gene_block_id].y += smemvec_d[gene_block_id + i].y;
        }
        __syncthreads();
    }
    
    if(threadIdx.x == 0){
        sum = smemvec_d[gene_block_id];

        f[chromo_id] = sqrt(fabs(sum.x*sum.x - sum.y*sum.y)) + (0.5*sum.x + sum.y)/nx + 0.5;
    }
} 

template<>
__global__ void hgbat_gpu<float>(float *x, float *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;

    extern __shared__ float2 smemvec_f[];

    float xi   = 0;
    float2 sum  = {0, 0};

    for(i = threadIdx.x; i < nx; i += blockDim.x){
        xi = x[chromo_id*nx + i] - 1.0;

        sum.x  += xi*xi;
        sum.y  += xi;
    }

    smemvec_f[gene_block_id] = sum;
    __syncthreads();
    
    for( i = blockDim.x / 2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            smemvec_f[gene_block_id].x += smemvec_f[gene_block_id + i].x;
            smemvec_f[gene_block_id].y += smemvec_f[gene_block_id + i].y;
        }
        __syncthreads();
    }
    
    if(threadIdx.x == 0){
        sum = smemvec_f[gene_block_id];

        f[chromo_id] = sqrt(fabs(sum.x*sum.x - sum.y*sum.y)) + (0.5*sum.x + sum.y)/nx + 0.5;
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

#ifndef ELLIPS_KERNEL_NO_CONSTANT
#define ELLIPS_KERNEL_NO_CONSTANT
template <typename T>
__global__ void ellips_gpu(T *x, T *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x; 

    extern __shared__ T s_mem[];

    T xi = 0;
    T value = 0;
    
    for(i = threadIdx.x; i < nx; i += blockDim.x){
        xi = x[chromo_id*nx + i];

        value += xi*xi*pow(10.0, 6.0*i/(nx-1));
    }

    s_mem[gene_block_id] = value;
    __syncthreads();
    
    reduction(gene_block_id, s_mem);

    if(threadIdx.x == 0){
        f[chromo_id] = s_mem[gene_block_id];
    }    
}
#endif

#ifndef BENTCIGAR_KERNEL_NO_CONSTANT
#define BENTCIGAR_KERNEL_NO_CONSTANT
template <typename T>
__global__ void bent_cigar_gpu(T *x, T *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;
    
    extern __shared__ T s_mem[];

    T xi = 0.0;
    T x1 = 0.0;
    T sum = 0.0;

    if(threadIdx.x < nx){
        xi = x[chromo_id*nx + threadIdx.x];
        x1 = xi; // first thread will keep this value
        sum = xi*xi;
    }

    for(i = blockDim.x+threadIdx.x; i < nx; i+= blockDim.x){
        xi = x[chromo_id*nx + i];
        sum += xi*xi;  
    }

    s_mem[gene_block_id] = sum;
    __syncthreads();

    reduction(gene_block_id, s_mem);

    if(threadIdx.x == 0){
        f[chromo_id] = 1e6*s_mem[gene_block_id] - 1e6*x1*x1 + x1*x1;
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