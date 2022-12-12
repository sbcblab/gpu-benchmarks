#include <stdio.h>
#include <random>
#include <type_traits>
#include <time.h>

#ifndef UTILS
#define UTILS
int next_pows(int n);
int read_dimensions(int argc, char *argv[]);
int read_population(int argc, char *argv[]);
int read_function(int argc, char *argv[]);
int read_shift(int argc, char *argv[]);
int read_rotate(int argc, char *argv[]);
int read_ipb(int argc, char *argv[]);

void print_time(clock_t t, const char msg[]);
void mul_matrix_vector_cpu(double *M, double *V, double *out, int n);
#endif

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

    } else if(std::is_same<T, int>::value){
        
        for(int i = 0; i < n-1; i++){
            printf("%d, ", x[i]);   
        }

        printf("%d ]\n", x[n-1]);      
    }
}
#endif