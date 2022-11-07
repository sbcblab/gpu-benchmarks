__global__ void shrink_vector(double *x, double *out, double shrink_rate, int n);
__global__ void shift_shrink_vector(double *X_dev, double *Opt_shift, double *out_dev, double shrink_rate, int n, int pop);
__global__ void mul_matrix_vector(double *M, double *V, double *out, int n);
