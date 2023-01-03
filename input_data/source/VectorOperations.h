#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <type_traits>
#include <math.h>
#include <time.h>
#include <errno.h>

#include "../utils.h"

#ifndef VECTOR_OPS
#define VECTOR_OPS


template <class T>
double magnitude(T *v, int n){
    double sum = 0;
    for(int i = 0; i < n; i++){
        sum += v[i]*v[i];
    }

    return sqrt(sum);
}

template <class T>
void normalize_vector(T *v, int n){
    double mag = magnitude<T>(v, n);

    for(int i = 0; i < n; i++){
        v[i] = v[i]/mag;
    }
}


template <class T>
T inner_product(T *u, T *v, int n){
    double sum = 0;
    for(int i = 0; i < n; i++){
        sum += u[i]*v[i];
    }

    return sum;
}

template <class T>
void projection(T *u, T* v, T* out, int n){
    double scalar = inner_product<T>(v, u, n)/inner_product<T>(u, u, n);

    for(int i = 0; i < n; i++){
        out[i] = scalar*u[i];
    }
}

template <class T>
void sub_vectors(T *u, T *v, T *out, int n){
    for(int i = 0; i < n; i++){
        out[i] = u[i] - v[i];
    }
}

template <class T>
void copy_vectors(T *in, T *out, int n){
    for(int i = 0; i < n; i++){
        out[i] = in[i];
    }
}


template <class T>
void check_orthonormality(T *rm, int n){

    for(int i = 0; i < n; i++){
        printf("mag v%d: %lf\n", i, magnitude<double>(&rm[i*n], n));           
       
        for(int j = i+1; j < n; j++){
            printf("<v%d, v%d>: %lf\n", i, j, inner_product<double>(&rm[i*n], &rm[j*n], n));
        }
    }

}

template <class T>
void transpose_matrix(T *m, T *out, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            out[j*n + i] = m[i*n + j];
        }
    }
}

template <class T>
void mul_matrix(T *a, T *b, T* out, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            out[i*n + j] = 0;
            for(int k = 0; k < n; k++){
                out[i*n +j] += a[i*n + k]*b[k*n +j];
            }
        }
    }
}

template <typename T>
void gram_schmidt(T *m, T *out, int n){
    
    T proj[n];
    T *init_out;

    copy_vectors<T>(&m[0], &out[0], n);
    normalize_vector<T>(&out[0], n);


    for(int i = 1; i < n; i++){
        copy_vectors<T>(&m[i*n], &out[i*n], n);

        for(int j = 0; j < i; j++){
            projection<T>(&out[j*n], &m[i*n], proj, n);
            sub_vectors<T>(&out[i*n], proj, &out[i*n], n);
        }
        
        normalize_vector<T>(&out[i*n], n);

    }
}

#endif 