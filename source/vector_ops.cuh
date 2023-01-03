#pragma once

#include <random>
#include <type_traits>

template <typename T>
__global__ void shrink_vector(T *x, T *out, float shrink_rate, int n);

template <typename T>
__global__ void shift_shrink_vector(T *X_dev, T *Opt_shift, T *out_dev, float shrink_rate, int n, int pop);

template <typename T>
__global__ void shuffle_vector(T *x, int*shuffled_indices, T *out, int n, int pop);

template <typename T>
__global__ void shuffle_vector(T *x, int *shuffled_indices, T *out, int n, int pop){
    int chromosome_id = blockIdx.x;

    __shared__ T smem[6144];

    x = &x[chromosome_id*n];

    for(int i = threadIdx.x; i < n; i += blockDim.x){
        smem[i] = x[i];
        
    }
    __syncthreads();
    
    for(int i = threadIdx.x; i < n; i += blockDim.x){
        
        x[i] = smem[shuffled_indices[i]]; 
    }


}

template <typename T>
__global__ void shrink_vector(T *x, T *out, float shrink_rate, int n){
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int id_dim = tid % n;   // dimensions id
    int id_ind = tid / n;   // individual id
    int index  = id_ind*n + id_dim;
    if(tid < n){
        out[index] = shrink_rate*x[index];
    }
}

// shrink_rate = FUNC_BOUND/X_BOUND 
template <typename T>
__global__ void shift_shrink_vector(T *X_dev, T *Opt_shift, T *out_dev, float shrink_rate, int n, int pop){
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    
    if(tid < n*pop){
        // shift vector and then shrink
        out_dev[tid] = shrink_rate*(X_dev[tid] - Opt_shift[tid%n]);
    }
}

#ifndef TEMPLATE_FUNC
#define TEMPLATE_FUNC
template <typename TYPE>
void init_random_square_matrix(TYPE *M, int n, int min, int max){
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());
    std::uniform_real_distribution<TYPE> dis(min, max);
    for(int i = 0; i < n*n; i++){
        M[i] = dis(gen);
    }
}

template <typename TYPE>
void init_random_vector(TYPE *V, int n, int min, int max){
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());
    std::uniform_real_distribution<TYPE> dis(min, max);
    for(int i = 0; i < n; i++){
        V[i] = dis(gen);
    }
}


template <typename TYPE>
void init_vector_from_binary(TYPE *V, int n, const char* filename){
    FILE *fp;

    fp = fopen(filename, "rb");

    if(!fp){
        perror("Error opening binary file for the rotation matrix");
        exit(EXIT_FAILURE);
    }

    if(V == NULL) {
        printf("Matrix pointer is a null pointer\n");
        exit(EXIT_FAILURE);
    }

    size_t ret = fread(V, n*sizeof(TYPE), 1, fp);

    fclose(fp);

}

template <typename TYPE>
void vector_to_file(TYPE *V, int n, const char* filename){
    FILE *fp;

    fp = fopen(filename, "w+");

    if(!fp){
        perror("Error opening binary file for the rotation matrix");
        exit(EXIT_FAILURE);
    }

    size_t ret = fwrite(V, n*sizeof(double), 1, fp);

}

template <typename TYPE>
void init_unitary_triangular_matrix(TYPE *M, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n-i; j++){
            M[i*n + j] = 1.0;
        }
        for(int j = n-i; j < n; j++){
            M[i*n + j] = 0.0;
        }
    }
}


template <class T>
void print_matrix(T *m, int n){
    for(int i = 0; i < n; i++){
        if(std::is_same<T, double>::value){
            for(int j = 0; j < n-1; j++){
                printf("%lf ", m[i*n + j]);
            }

            printf("%lf\n", m[i*n + n-1]);    
        }
        else if(std::is_same<T, float>::value){
            for(int j = 0; j < n-1; j++){
                printf("%f ", m[i*n + j]);
            }

            printf("%f\n", m[i*n + n-1]);  
        }
    }
}

template <class T>
void print_vector(T *x, int n){

    printf("[ ");
    
    if(std::is_same<T, double>::value){
        for(int i = 0; i < n-1; i++){
            printf("%lf, ", x[i]);   
        }

        printf("%lf ]\n", x[n-1]);   
    } 
    else if(std::is_same<T, float>::value){
        for(int i = 0; i < n-1; i++){
            printf("%f, ", x[i]);   
        }

        printf("%f ]\n", x[n-1]);  

    }
}

template <class T>
inline __device__ T fmod_device(T a, T b);

template<>
inline __device__ double fmod_device<double>(double a, double b){
    return fmod(a, b);
}


template<>
inline __device__ float fmod_device<float>(float a, float b){
    return fmodf(a, b);
}
#endif