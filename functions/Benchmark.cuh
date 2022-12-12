#pragma once

#include "cublas_v2.h"
#include "../utils.h"
#include "../gpu_constants.cuh"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ __constant__ float p_lambda_dev[8];
__device__ __constant__ float p_bias_dev[8];
__device__ __constant__ float p_delta_dev[8];


template<typename T>
__global__ void cfcal_gpu(T *x, T *f, T *Os, T *fit, int nx);

template <class T> 
class Benchmark {
    protected:
        T *p_aux_dev; T *rot_dev;
        T *p_cfit_dev;
        T *p_x_dev = NULL; T *p_f_dev = NULL; 
        T *p_rotm_dev; T *p_shift_dev;
        int *p_shuffle_dev;

        cublasHandle_t handle;

        int n; 
        int pop_size;
        int func_num;
        bool user_device_input_pointer = false;
        bool user_device_output_pointer = false;
        bool shift_func = false;
        bool rot_func = false;
        bool shuffle_func = false;
        
        T func_boundary;
        
        dim3 block_shape;
        int grid_size;
        int shared_mem_size;
        int grid_size_shift;

        int individuals_per_block(int pop_size, int threads_p_individual){
            return max( 1,              // if division equals to zero, use 1 individual per block instead
                        min(pop_size, MIN_OCCUPANCY/ threads_p_individual));
        }

        int max_threads_per_individual(){
            return min( MAX_BLOCK_SIZE, 
                        next_pows(n)); // use a power of two threads because of the parallel reduction
        }

        void kernel_launch_config(int &grid_size, dim3 &block_size, int &shared_mem_size){
            int threads_p_individual = max_threads_per_individual();
            int chromosomes_p_block  = individuals_per_block(pop_size, threads_p_individual);
            
            grid_size       = pop_size/chromosomes_p_block;
            block_size.x    = threads_p_individual;
            block_size.y    = chromosomes_p_block;
            shared_mem_size = chromosomes_p_block*threads_p_individual*sizeof(double);
    
        }

        // set global optimum
        void use_shift_vector(const char *filename, int count_elements){
            T* V;

            V = (T*)malloc(sizeof(T)*count_elements);

            cudaMalloc<T>(&p_shift_dev, count_elements*sizeof(T));

            init_vector_from_binary<T>(V, count_elements, filename);
            cudaMemcpy(p_shift_dev, V, count_elements*sizeof(T), cudaMemcpyHostToDevice);

            // set shift flag
            shift_func = true;

            free(V);
        }

        void use_rotation_matrix(const char *filename, int count_elements){
            T* M;

            M = (T*)malloc(sizeof(T)*count_elements);
            
            // matrix pointer
            cudaMalloc<T>(&p_rotm_dev, count_elements*sizeof(T));
            
            // rotated solutions pointer
            cudaMalloc<T>(&rot_dev, n*pop_size*sizeof(T));

            init_vector_from_binary<T>(M, count_elements, filename);                      
            cudaMemcpy(p_rotm_dev, M, count_elements*sizeof(T), cudaMemcpyHostToDevice);

            // set rotation flag    
            rot_func = true;

            free(M);

        }

        void use_shuffle_vector(const char *filename, int count_elements){
            int *S;

            S = (int*)malloc(sizeof(int)*count_elements);
            cudaMalloc<int>(&p_shuffle_dev, count_elements*sizeof(int));

            init_vector_from_binary<int>(S, count_elements, filename);

            cudaMemcpy(p_shuffle_dev, S, count_elements*sizeof(int), cudaMemcpyHostToDevice);

            shuffle_func = true;

            free(S);
        }

        void checkPointers(T *x, T *f){
            cudaPointerAttributes input_attributes;
            cudaPointerAttributes output_attributes;

            cudaPointerGetAttributes(&input_attributes, x);
            cudaPointerGetAttributes(&output_attributes, f);

            switch(input_attributes.type){
                case cudaMemoryType::cudaMemoryTypeHost: 
                case cudaMemoryType::cudaMemoryTypeUnregistered:{

                    if(p_x_dev == NULL){
                        printf("Allocating memory on device for the input pointer\n");
                        cudaMalloc<T>(&p_x_dev, sizeof(T)*pop_size*n);
                    }
                    
                    cudaMemcpy(p_x_dev, x, sizeof(T)*pop_size*n, cudaMemcpyHostToDevice);
                    
                    break;
                }
                case cudaMemoryType::cudaMemoryTypeDevice: {
                    p_x_dev = x;
                    user_device_input_pointer = true;
                    break;
                }
            }

            switch(output_attributes.type){
                case cudaMemoryType::cudaMemoryTypeHost: 
                case cudaMemoryType::cudaMemoryTypeUnregistered: {
                    
                    printf("Allocating memory on device for the output pointer\n");
                    if(p_f_dev == NULL){
                        cudaMalloc<T>(&p_f_dev, sizeof(T)*pop_size);
                    }
                    break;
                }
                case cudaMemoryType::cudaMemoryTypeDevice: {
                    p_f_dev = f;
                    user_device_output_pointer = true;
                    break;
                }
            }

        }

        void checkOutput(T *f){
            cudaPointerAttributes output_attributes;

            cudaPointerGetAttributes(&output_attributes, f);

            switch(output_attributes.type){
                case cudaMemoryType::cudaMemoryTypeHost: 
                case cudaMemoryType::cudaMemoryTypeUnregistered: {
                    cudaMemcpy(f, p_f_dev, sizeof(T)*pop_size, cudaMemcpyDeviceToHost);
                    break;
                }
                case cudaMemoryType::cudaMemoryTypeDevice: {
                    break;
                }
            }

        }

        void freeIO(){
            if(p_x_dev != NULL && user_device_input_pointer == false){
                cudaFree(p_x_dev);
            }
            
            if(p_f_dev != NULL && user_device_output_pointer == false){
                cudaFree(p_f_dev);
            }
        }

        void rotation(T *rot_matrix){

        }

        virtual void allocateMemory(){
            // empty
        }

        virtual void freeMemory(){
            // empty
        }


    public:
        Benchmark(){
            n = 32;
            pop_size = 32;
        }
        virtual ~Benchmark(){

        }

        virtual void compute(T *p_x_dev, T*p_f_dev)
        {
            /* empty */
        };


        // void input(T *x){
        //     cudaMemcpy(p_x_dev, x, n*pop_size*sizeof(T), cudaMemcpyHostToDevice);
        // }

        // void output(T *f){
        //     cudaMemcpy(f, p_f_dev, pop_size*sizeof(T), cudaMemcpyDeviceToHost);
        // }
        
        // T* get_input_dev(){
        //     return p_x_dev;
        // }

        // T* get_output_dev(){
        //     return p_f_dev;
        // }

        uint getID();

        void setMin( T );
        void setMax( T );

        /* GPU launch compute status */
        void set_launch_config(int grid, dim3 block){
            grid_size = grid;
            block_shape = block;
        }

        dim3 getBlockSize(){
            return block_shape;
        }

        int getGridSize(){
            return grid_size;
        }


};


template<> 
void Benchmark<double>::rotation(double *rot_matrix){
    double alpha = 1.0;
    double beta  = 0.0;

    cublasDgemm( handle, CUBLAS_OP_T, CUBLAS_OP_N, n, pop_size, n, &alpha, rot_matrix, n, p_aux_dev, n, &beta, rot_dev, n );
}

template<> 
void Benchmark<float>::rotation(float *rot_matrix){
    float alpha = 1.0;
    float beta  = 0.0;

    cublasSgemm( handle, CUBLAS_OP_T, CUBLAS_OP_N, n, pop_size, n, &alpha, rot_matrix, n, p_aux_dev, n, &beta, rot_dev, n );
}

template<typename T>
__global__ void cfcal_gpu(T *x, T *f, T *Os, T *fit, int nx){
    int i;
    cg::thread_block_tile<WARP_SIZE> group = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

    __shared__ T smem[WARP_SIZE];

    T xi;
    T w = 0.0;
    T f_temp = 0.0;        

    x = &x[blockIdx.x*nx];
    Os = &Os[threadIdx.y*nx];
    fit = &fit[blockIdx.x*blockDim.y];


    for(i = group.thread_rank(); i < nx; i += group.size()){
        xi = x[i] - Os[i];
        w += xi*xi;
    }

    w += group.shfl_down(w, 16);
    w += group.shfl_down(w, 8);
    w += group.shfl_down(w, 4);
    w += group.shfl_down(w, 2);
    w += group.shfl_down(w, 1);

    if(group.thread_rank() == 0){
        smem[threadIdx.y] = w;
    }
    __syncthreads();

    T w_sum = 0.0;
    T w_temp = 0.0;

    if(group.meta_group_rank() == 0){
        w = 0;
        if(group.thread_rank() < blockDim.y){
            w = smem[group.thread_rank()];
            
            float d = p_delta_dev[group.thread_rank()];
            float b = p_bias_dev[group.thread_rank()];
            
            f_temp = fit[group.thread_rank()]*p_lambda_dev[group.thread_rank()] + b;
            if(w != 0){
                w = pow(1.0/w,0.5)*exp(-w/2.0/nx/pow(d, 2.0));
            } else {
                w = INFINITY;
            }

        }

        w_sum = w;
        w_sum += group.shfl_down(w_sum, 4);
        w_sum += group.shfl_down(w_sum, 2);
        w_sum += group.shfl_down(w_sum, 1);
        
        w_temp = f_temp*w;

        w_temp += group.shfl_down(w_temp, 4);
        w_temp += group.shfl_down(w_temp, 2);
        w_temp += group.shfl_down(w_temp, 1);

        if(group.thread_rank() == 0){
            f[blockIdx.x] = w_temp/w_sum;
        }
    }
}
