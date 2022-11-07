
#ifndef BENCH_GPU
#define BENCH_GPU


__device__ void reduction( int index, double *s_mem );

__global__ void mul_matrix_vector(double *M, double *V, double *out, int n);
__global__ void mul_matrix_vector_2(double *M, double *V, double *out, int n);
__global__ void shift_shrink_vector(double *X_dev, double *Opt_shift, double *out_dev, double shrink_rate, int n, int pop);
__global__ void shrink_vector(double *x, double *out, double shrink_rate, int n);
 
__global__ void zakharov_gpu(double *x, double *f, int nx); 
__global__ void rastrigin_gpu(double *x, double *f, int nx); // correct
__global__ void schaffer_F7_gpu(double *x, double *f, int nx); // i don't know if CEC's implementation is correct
__global__ void rosenbrock_gpu(double *x, double *f, int nx); // correct
__global__ void bent_cigar_gpu(double *x, double *f, int nx); // correct; precision error
__global__ void hgbat_gpu(double *x, double *f, int nx); // correct
__global__ void schaffer_F6_gpu(double *x, double *f, int nx);
__global__ void ackley_gpu(double *x, double *f, int nx); // correct
__global__ void levy_gpu(double *x, double *f, int nx); // correct
__global__ void ellips_gpu(double *x, double *f, int nx); // correct
__global__ void happycat_gpu(double *x, double *f, int nx); // correct
__global__ void discus_gpu(double *x, double *f, int nx); // correct; precision error
__global__ void griewank_gpu(double *x, double *f, int nx); // correct
__global__ void katsuura_gpu(double *x, double *f, int nx); // correct
__global__ void grie_rosen_gpu(double *x, double *f, int nx); // correct; sometimes get precision errors
__global__ void schwefel_gpu(double *x, double *f, int nx);   // correct
__global__ void escaffer6_gpu(double *x, double *f, int nx); // correct

__global__ void hf02_gpu(double *x, double *f, int nx);
__global__ void hf10_gpu(double *x, double *f, int nx);
__global__ void hf06_gpu(double *x, double *f, int nx);

__global__ void cf_cal_gpu(double *x, double *f, double *Os, double *lambda, double *delta, double *bias, double *fit, int nx);
#endif