/*
*   Create a random rotation matrix using the 
*   Gram-Schmidt orthonormalization process
*/


typedef enum {
    TXT_FILE,
    BIN_FILE
} file_type;


void rotation_matrix(double *m, int n);
void rotation_matrix(float *m,  int n);
void rotation_matrix(float *m, int n, const char* filename, file_type F_TYPE);
void rotation_matrix(double *m, int n, const char* filename, file_type F_TYPE);
void rotation_matrix_from_file(float *m, int n, const char* filename, file_type F_TYPE);
