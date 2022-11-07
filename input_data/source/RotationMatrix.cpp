#include "VectorOperations.h"
#include "RotationMatrix.h"

void rotation_matrix(double *m, int n){
    double *rand_M, *aux_M;

    rand_M = (double *)malloc(n*n*sizeof(double));
    aux_M =  (double *)malloc(n*n*sizeof(double));

    for(int i = 0; i < n; i++){
        init_random_vector<double>(&rand_M[i*n], n, 0, 1);
    }

    gram_schmidt<double>(rand_M, aux_M, n);
    transpose_matrix<double>(aux_M, m, n);

    free(aux_M);
    free(rand_M);
}

void rotation_matrix(float *m, int n){
    float *rand_M, *aux_M;

    rand_M = (float *)malloc(n*n*sizeof(float));
    aux_M =  (float *)malloc(n*n*sizeof(float));

    for(int i = 0; i < n; i++){
        init_random_vector<float>(&rand_M[i*n], n, 0, 1);
    }

    gram_schmidt<float>(rand_M, aux_M, n);
    transpose_matrix<float>(aux_M, m, n);

    free(aux_M);
    free(rand_M);
}


/*
* Create a random rotation matrix and save to file
*/
void rotation_matrix(float *m, int n, const char* filename, file_type F_TYPE){

    FILE *ptr;

    if(F_TYPE == TXT_FILE){
        
    } else if(F_TYPE == BIN_FILE) {
        ptr = fopen(filename, "wb");
        
        if(!ptr){
            perror("Error creating binary file for the rotation matrix");
            exit(EXIT_FAILURE);
        }

        // create rotation matrix
        if(m == NULL) {
            printf("Matrix pointer is a null pointer\n");
            exit(EXIT_FAILURE);
        }
        rotation_matrix(m, n);

        fwrite(m, n*sizeof(float), n, ptr);

    }
    
    fclose(ptr);
}

void rotation_matrix(double *m, int n, const char* filename, file_type F_TYPE){

    FILE *ptr;

    if(F_TYPE == TXT_FILE){
        
    } else if(F_TYPE == BIN_FILE) {
        ptr = fopen(filename, "wb");
        
        if(!ptr){
            perror("Error creating binary file for the rotation matrix");
            exit(EXIT_FAILURE);
        }

        // create rotation matrix
        if(m == NULL) {
            printf("Matrix pointer is a null pointer\n");
            exit(EXIT_FAILURE);
        }
        rotation_matrix(m, n);

        fwrite(m, n*sizeof(float), n, ptr);

    }
    
    fclose(ptr);
}

/*
* Read file containing a matrix 
*/
void rotation_matrix_from_file(float *m, int n, const char* filename, file_type F_TYPE){
    FILE *ptr;

    if(F_TYPE == TXT_FILE){
        
    } else if(F_TYPE == BIN_FILE) {
        ptr = fopen(filename, "rb");

        if(!ptr){
            perror("Error opening binary file for the rotation matrix");
            exit(EXIT_FAILURE);
        }

        // create rotation matrix
        if(m == NULL) {
            printf("Matrix pointer is a null pointer\n");
            exit(EXIT_FAILURE);
        }

        fread(m, n*sizeof(float), n, ptr);

    }    

    fclose(ptr);
}