#include <stdlib.h>
#include <stdio.h>
#include <random>
// #include <math.h>
#include <time.h>
#include <string.h>
#include <cooperative_groups.h>

#include "gpu_constants.cuh"
#include "benchmark_constants.cuh"
#include "benchmark_kernels.cuh"

#define E  2.7182818284590452353602874713526625
#define PI 3.1415926535897932384626433832795029

namespace cg = cooperative_groups;


__device__ void reduction_mult( int index, double *s_mem ){
    int i;
    for( i = blockDim.x / 2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            s_mem[index] *= s_mem[index + i];
        }
        __syncthreads();
    }
}

__device__ void reduction( int index, double *s_mem ){
    int i;

    for( i = blockDim.x / 2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            s_mem[index] += s_mem[index + i];
        }
        __syncthreads();
    }
    
}

__device__ double g_schaffer_f6(double x, double y){
    double num = sin(sqrt(x*x + y*y))*sin(sqrt(x*x + y*y)) - 0.5;
    double dem = (1 + 0.001*(x*x + y*y))*(1 + 0.001*(x*x + y*y));
    return 0.5 + num/dem;
}

__device__ double w_levy(double x){
    return 1 + (x - 0.0)/4.0;
}

__global__ void schaffer_F6_gpu(double *x, double *f, int nx){

}

__global__ void zakharov_gpu(double *x, double *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;

    extern __shared__ double2 smemvec[];

    double2 sum = {0, 0};

    double value = 0;
    
    for(i = threadIdx.x; i < nx; i += blockDim.x){
        value = x[chromo_id*nx + i];

        sum.x += value*value;
        sum.y += 0.5*(i+1)*value;
    }

    smemvec[gene_block_id] = sum;
    __syncthreads();
    
    for( i = blockDim.x / 2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            smemvec[gene_block_id].x += smemvec[gene_block_id + i].x;
            smemvec[gene_block_id].y += smemvec[gene_block_id + i].y;
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        sum = smemvec[gene_block_id];
        sum.y = sum.y*sum.y;
        f[chromo_id] = sum.x + sum.y + sum.y*sum.y; 
    }

}

__global__ void rastrigin_gpu(double *x, double *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;

    extern __shared__ double s_mem[];

    double xi = 0;
    double value = 0;
    
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

__global__ void rosenbrock_gpu(double *x, double *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;
    
    extern __shared__ double s_mem[];

    double xi = 0;
    double x_next = 0;
    double sum = 0;

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


__global__ void ackley_gpu(double *x, double *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;

    extern __shared__ double2 smemvec[];

    double xi = 0;
    double2 sum = {0, 0};

    for(i = threadIdx.x; i < nx; i += blockDim.x){
        xi = x[chromo_id*nx + i];

        sum.x += xi*xi;
        sum.y += cos(2*PI*xi);
    }

    smemvec[gene_block_id] = sum;
    __syncthreads();

    for( i = blockDim.x / 2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            smemvec[gene_block_id].x += smemvec[gene_block_id + i].x;
            smemvec[gene_block_id].y += smemvec[gene_block_id + i].y;
        }
        __syncthreads();
    }
 

    if(threadIdx.x == 0){
        sum = smemvec[gene_block_id];
        double c = 1.0 / double(nx);

        f[chromo_id] = 20 - 20*exp(-0.2*sqrt( c * sum.x)) + E - exp(c * sum.y) ;  
    }
    
    
}

__global__ void step_rastrigin_gpu(double *x, double *f, int nx){

}

__global__ void bent_cigar_gpu(double *x, double *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;
    
    extern __shared__ double s_mem[];

    double xi = 0.0;
    double x1 = 0.0;
    double sum = 0.0;

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

__global__ void hgbat_gpu(double *x, double *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;

    extern __shared__ double2 smemvec[];

    double xi   = 0;
    double2 sum  = {0, 0};

    for(i = threadIdx.x; i < nx; i += blockDim.x){
        xi = x[chromo_id*nx + i] - 1.0;

        sum.x  += xi*xi;
        sum.y  += xi;
    }

    smemvec[gene_block_id] = sum;
    __syncthreads();
    
    for( i = blockDim.x / 2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            smemvec[gene_block_id].x += smemvec[gene_block_id + i].x;
            smemvec[gene_block_id].y += smemvec[gene_block_id + i].y;
        }
        __syncthreads();
    }
    
    if(threadIdx.x == 0){
        sum = smemvec[gene_block_id];

        f[chromo_id] = sqrt(fabs(sum.x*sum.x - sum.y*sum.y)) + (0.5*sum.x + sum.y)/nx + 0.5;
    }
} 

__global__ void happycat_gpu(double *x, double *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;

    extern __shared__ double s_mem[];

    double xi   = 0;
    double sum  = 0;
    double sum2 = 0;

    for(i = threadIdx.x; i < nx; i += blockDim.x){
        xi = x[chromo_id*nx + i] - 1.0;

        sum  += xi*xi;
        sum2 += xi;
    }

    s_mem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, s_mem);
    
    if(threadIdx.x == 0){
        sum = s_mem[gene_block_id];
    }

    s_mem[gene_block_id] = sum2;
    __syncthreads();
    reduction(gene_block_id, s_mem);

    if(threadIdx.x == 0){
        sum2 = s_mem[gene_block_id];

        f[chromo_id] = pow(fabs(sum - nx), 0.25) + (0.5*sum + sum2)/nx + 0.5;
    }
} 

__global__ void escaffer6_gpu(double *x, double *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;
    
    extern __shared__ double s_mem[];

    double xi   = 0;
    double xi_1 = 0;
    double sum  = 0;
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

__global__ void schaffer_F7_gpu(double *x, double *f, int nx){
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;
    // cg::thread_block_tile<WARP_SIZE> group = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());


    extern __shared__ double s_mem[];

    double sum = 0.0;

    s_mem[gene_block_id] = 0.0;
    __syncthreads(); 

        
    for(int i = threadIdx.x; i < nx - 1; i += blockDim.x){
        // talvez a cache ajude nessa operação
        double si = pow(x[chromo_id*nx + i]*x[chromo_id*nx + i] + x[chromo_id*nx + i +1]*x[chromo_id*nx + i + 1], 0.5);
        sum += pow(si, 0.5) + pow(si, 0.5)*sin(50.0*pow(si, 0.2))*sin(50.0*pow(si, 0.2));
        
    }

    s_mem[gene_block_id] = sum;
    __syncthreads();

    reduction(gene_block_id, s_mem);

    if(threadIdx.x == 0){

        f[chromo_id] = s_mem[gene_block_id]*s_mem[gene_block_id]/((nx-1)*(nx-1));
    }
    
}

__global__ void levy_gpu(double *x, double *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;

    extern __shared__ double s_mem[];

    double wi   = 0.0;
    double sum = 0.0;

    if(threadIdx.x < nx){
        wi = w_levy(x[chromo_id*nx + threadIdx.x]);

        if(threadIdx.x == 0){
            sum = sin(PI*wi)*sin(PI*wi);
        }
        
        if(threadIdx.x == nx - 1){
            sum = (wi - 1)*(wi - 1)*(1 + sin(2*PI*wi)*sin(2*PI*wi));
        } else {
            sum += pow((wi-1),2) * (1+10*pow((sin(PI*wi+1)),2));
        }
    }


    for(i = threadIdx.x + blockDim.x; i < nx; i += blockDim.x){
        wi = w_levy(x[chromo_id*nx + i]);

        if(i == nx - 1){
            sum += (wi - 1)*(wi - 1)*(1 + sin(2*PI*wi)*sin(2*PI*wi));
        } else {
            sum += pow((wi-1),2) * (1+10*pow((sin(PI*wi+1)),2));
        }
    }


    s_mem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, s_mem);

    if(threadIdx.x == 0){
        f[chromo_id] = s_mem[gene_block_id];
    }

}

__global__ void ellips_gpu(double *x, double *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x; 

    extern __shared__ double s_mem[];

    double xi = 0;
    double value = 0;
    
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

__global__ void griewank_gpu(double *x, double *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x; 

    extern __shared__ double s_mem[];

    double xi;

    double term1 = 0.0;
    double term2 = 1.0;

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

__global__ void katsuura_gpu(double *x, double *f, int nx){
    int i;
    uint j;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;
    
    extern __shared__ double s_mem[];

    double xi = 0.0;
    double term = 1.0;

    for(i = threadIdx.x; i < nx; i += blockDim.x){
        double sumJ = 0.0;

        xi = x[chromo_id*nx + i];

        for(j = 1; j <= 32; j ++){
            sumJ += fabs(pow(2.0,j)*xi - round(pow(2.0,j)*xi))/pow(2.0,j); 
        }

        term *= pow( 1.0+(i+1)*sumJ , 10.0/pow(nx, 1.2) );
    }



    s_mem[gene_block_id] = term;
    __syncthreads();
    reduction_mult(gene_block_id, s_mem);

    if(threadIdx.x == 0){
        double v = s_mem[gene_block_id];
        f[chromo_id] = (10.0/(nx*nx))*v - 10.0/(nx*nx);
    }
}

__global__ void grie_rosen_gpu(double *x, double *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;
    
    extern __shared__ double s_mem[];

    double xi = 0.0;
    double x_next = 0.0;

    double temp = 0.0;
    double sum  = 0.0;

    double x1 = 0.0;
    for(i = threadIdx.x; i < nx; i += blockDim.x){
        xi = x[chromo_id*nx + i] + 1;
        
        
        if(i == (nx-1)%blockDim.x){
            x1 = x[chromo_id*nx];
        }

        if(i == nx-1){
            x_next = x1 +1;
        } else {
            x_next = x[chromo_id*nx + (i+1)] +1;
        }
        
        temp = 100*pow((xi*xi - x_next), 2) + (xi-1)*(xi-1); // rosenbrock(x1, x2)
        temp = temp*temp/4000 - cos(temp) + 1; // griewank(x1)
        
        sum += temp;
    } 

    s_mem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, s_mem);

    if(threadIdx.x == 0){
        f[chromo_id] = s_mem[gene_block_id]; 
    }
}

__global__ void schwefel_gpu(double *x, double *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;
    
    extern __shared__ double s_mem[];

    double sum = 0.0;
    double zi;
    for(i = threadIdx.x ; i < nx; i += blockDim.x){
        zi = 4.209687462275036e+002 + x[chromo_id*nx + i];
        double zi_fmod = fmod(zi, 500.0);
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

__global__ void discus_gpu(double *x, double *f, int nx){
    int i;
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;
    
    extern __shared__ double s_mem[];

    double xi = 0.0;
    double x1 = 0.0;
    double sum = 0.0;

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
        f[chromo_id] = s_mem[gene_block_id] - x1*x1 + 1e6*x1*x1;
    }

}

__global__ void hf02_gpu(double *x, double *f, int nx){
    #define HF02_n1 0.4
    #define HF02_n2 0.4
    #define HF02_n3 0.2
    
    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;

    extern __shared__ double smem[];

    double xi, x1;
    double sum  = 0.0;
    double sum2 = 0.0;
    double fit;

    int i = threadIdx.x;
    int n_f = ceil(nx*HF02_n1);

    x = &x[chromo_id*nx];

    // bent-cigar

    if( i < n_f ){
        xi = x[i];
        x1 = xi;
        sum = xi*xi;
    }

    i += blockDim.x;
    for(; i < n_f; i += blockDim.x){
        xi = x[i];
        sum += xi*xi;
    }

    smem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, smem);
    
    if(threadIdx.x == 0){
        fit = 1e6*smem[gene_block_id] - 1e6*x1*x1 + x1*x1;
    }


    // hg-bat
    i = threadIdx.x + n_f;
    n_f += ceil(nx*HF02_n2);
    for(; i < n_f; i += blockDim.x){
        xi = x[i] - 1.0;

        sum += xi*xi;
        sum2 += xi;
    }

    smem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        sum = smem[gene_block_id];
    }

    smem[gene_block_id] = sum2;
    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        sum2 = smem[gene_block_id];
        fit += sqrt(fabs(sum*sum - sum2*sum2)) + (0.5*sum + sum2)/nx + 0.5;
    }

  
    // rastrigin
    i = threadIdx.x + n_f;
    sum = 0.0;
    for(; i < nx; i += blockDim.x){
        xi = x[i];

        sum += xi*xi - 10*cos(2*PI*xi) + 10;
    }

    smem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        f[chromo_id] = smem[gene_block_id] + fit;
    }
}

__global__ void hf10_gpu(double *x, double *f, int nx){
    #define HF10_n1 0.2
    #define HF10_n2 0.2
    #define HF10_n3 0.2
    #define HF10_n4 0.2
    #define HF10_n5 0.1

    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;

    extern __shared__ double smem[];

    double xi;
    double sum = 0.0;
    double sum2 = 0.0;
    double fit;

    x = &x[chromo_id*nx];

    // HGBAT
    int i = threadIdx.x;
    int n_f = ceil(nx*HF10_n1);
    for(; i < n_f; i += blockDim.x){
        xi = x[i]*HGBAT_BOUND/X_BOUND - 1.0;

        sum += xi*xi;
        sum2 += xi;
    }

    smem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        sum = smem[gene_block_id];
    }
    
     smem[gene_block_id] = sum2;
    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        sum2 = smem[gene_block_id];
        fit = sqrt(fabs(sum*sum - sum2*sum2)) + (0.5*sum + sum2)/ceil(nx*HF10_n1) + 0.5;
    }

    // KATSUURA
    i = threadIdx.x + n_f;
    n_f += (int)ceil(nx*HF10_n2);
    sum = 1.0; // a multiplication is computed
    for(; i < n_f; i += blockDim.x){
        double sumJ = 0.0;

        xi = x[i]*KATSUURA_BOUND/X_BOUND;

        for(int j = 1; j <= 32; j ++){
            sumJ += fabs(pow(2.0,j)*xi - round(pow(2.0,j)*xi))/pow(2.0,j); 
        }

        sum *= pow( 1.0+((i+1 - (n_f - ceil(nx*HF10_n2))))*sumJ , 10.0/pow(ceil(nx*HF10_n2), 1.2) );

    }

    
    smem[gene_block_id] = sum;
    __syncthreads();
    reduction_mult(gene_block_id, smem);

    if(threadIdx.x == 0){
        fit += (10.0/(ceil(nx*HF10_n2)*ceil(nx*HF10_n2)))*smem[gene_block_id] - 10.0/(ceil(nx*HF10_n2)*ceil(nx*HF10_n2));
    }

    // ACKLEY
    i = threadIdx.x + n_f;
    n_f += ceil(nx*HF10_n3);
    sum = 0.0;
    sum2 = 0.0;
    for(; i < n_f; i += blockDim.x){
        xi = x[i]*ACKLEY_BOUND/X_BOUND;

        sum += xi*xi;
        sum2 += cos(2*PI*xi);
    }

    smem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        sum = smem[gene_block_id];
    }
    
    smem[gene_block_id] = sum2;
    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        sum2 = smem[gene_block_id];
        double c = 1.0 / double(ceil(nx*HF10_n3));

        fit  += 20 - 20*exp(-0.2*sqrt( c * sum)) + E - exp(c * sum2) ;  
    }


    // RASTRIGIN
    i = threadIdx.x + n_f;
    n_f += ceil(nx*HF10_n4);
    sum = 0.0;
    for(; i < n_f; i += blockDim.x){
        xi = x[i]*RASTRIGIN_BOUND/X_BOUND;

        sum += xi*xi - 10*cos(2*PI*xi) + 10;
    }

    smem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        fit += smem[gene_block_id] ;
    }

    // SCHWEFEL
    i = threadIdx.x + n_f;
    n_f += ceil(nx*HF10_n5);
    sum = 0.0;
    double zi;
    for(; i < n_f; i += blockDim.x){
        zi = 4.209687462275036e+002 + x[i]*SCHWEFEL_BOUND/X_BOUND;
        double zi_fmod = fmod(zi, 500.0);
        if(fabs(zi) <= 500.0){
            sum += zi*sin(pow(fabs(zi),0.5));
        } else if(zi > 500.0) {
            sum += (500 - zi_fmod)*sin(pow(fabs(500 - zi_fmod), 0.5)) - (zi - 500)*(zi - 500)/(10000*ceil(nx*HF10_n5));
        } else{
            sum += (-500.0+fmod(fabs(zi),500.0))*sin(pow(500.0-fmod(fabs(zi),500.0),0.5)) - (zi + 500)*(zi + 500)/(10000*ceil(nx*HF10_n5));
        }
    }

    smem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        fit += 4.189828872724338e+002 * ceil(nx*HF10_n5) - smem[gene_block_id];
    }

    // SCHAFFER F7
    i = threadIdx.x + n_f;
    sum = 0.0;
    for(; i < nx - 1; i += blockDim.x){
        // talvez a cache ajude nessa operação
        
        xi = x[i]*SCHAFFER_F7_BOUND/X_BOUND;
        double x_next = x[i + 1]*SCHAFFER_F7_BOUND/X_BOUND;

        double si = pow(xi*xi + x_next*x_next, 0.5);
        sum += pow(si, 0.5) + pow(si, 0.5)*sin(50.0*pow(si, 0.2))*sin(50.0*pow(si, 0.2));
        
    }

    smem[gene_block_id] = sum;
    __syncthreads();

    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){

        f[chromo_id] = smem[gene_block_id]*smem[gene_block_id]/(((nx-n_f)-1)*((nx-n_f)-1)) + fit;
    }
}

__global__ void hf06_gpu(double *x, double *f, int nx){
    #define HF06_n1 0.3
    #define HF06_n2 0.2
    #define HF06_n3 0.2
    #define HF06_n4 0.1

    int chromo_id = blockIdx.x*blockDim.y + threadIdx.y;
    int gene_block_id   = threadIdx.y*blockDim.x + threadIdx.x;

    extern __shared__ double smem[];

    double xi;
    double sum = 0.0;
    double sum2 = 0.0;
    double fit;

    x = &x[chromo_id*nx];

    // KATSUURA
    int i = threadIdx.x;
    int n_f = ceil(nx*HF06_n1);
    sum = 1.0; // a multiplication is computed
    for( ; i < n_f; i += blockDim.x){
        double sumJ = 0.0;

        xi = x[i]*KATSUURA_BOUND/X_BOUND;

        for(int j = 1; j <= 32; j ++){
            sumJ += fabs(pow(2.0,j)*xi - round(pow(2.0,j)*xi))/pow(2.0,j); 
        }

        sum *= pow( 1.0+(i+1)*sumJ , 10.0/pow(n_f, 1.2) );
    }


    smem[gene_block_id] = sum;
    __syncthreads();
    reduction_mult(gene_block_id, smem);

    if(threadIdx.x == 0){
        fit = (10.0/(n_f*n_f))*smem[gene_block_id] - 10.0/(n_f*n_f);

    }

    // HAPPYCAT
    i = threadIdx.x + n_f;
    n_f += ceil(nx*HF06_n2);
    sum = 0.0;
    sum2 = 0.0;    
    for( ; i < n_f; i += blockDim.x){

        xi = x[i]*HAPPYCAT_BOUND/X_BOUND - 1.0;

        sum  += xi*xi;
        sum2 += xi;
    }

    smem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, smem);
    
    if(threadIdx.x == 0){
        sum = smem[gene_block_id];
    }

    smem[gene_block_id] = sum2;
    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        sum2 = smem[gene_block_id];

        fit += pow(fabs(sum - ceil(nx*HF06_n2)), 0.25) + (0.5*sum + sum2)/ceil(nx*HF06_n2) + 0.5;

    }

    // GRIEROSEN
    i = threadIdx.x + n_f;
    double x1 = x[n_f]*GRIE_ROSEN_BOUND/X_BOUND + 1;
    n_f += ceil(nx*HF06_n3);
    double x_next = 0.0;
    sum = 0.0;
    for(; i < n_f; i += blockDim.x){
        xi = x[i]*GRIE_ROSEN_BOUND/X_BOUND + 1;
        // if(i == n_f-1){
        //     x1 = x[i - ((int)ceil(nx*HF06_n3)-1)]*GRIE_ROSEN_BOUND/X_BOUND + 1;    
        // }

        if(i == n_f-1){
            x_next = x1;
        } else {
            x_next = x[i+1]*GRIE_ROSEN_BOUND/X_BOUND  +1;
        }

        sum2 = 100*pow((xi*xi - x_next), 2) + (xi-1)*(xi-1); // rosenbrock(x1, x2)
        sum2 = sum2*sum2/4000 - cos(sum2) + 1; // griewank(x1)
        sum += sum2;
    } 
    smem[gene_block_id] = sum;

    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        fit += smem[gene_block_id]; 
    }

    // SCHWEFEL
    i = threadIdx.x + n_f;
    n_f += ceil(nx*HF06_n4);
    sum = 0.0;
    double zi;
    for( ; i < n_f; i += blockDim.x){
        zi = 4.209687462275036e+002 + x[i]*SCHWEFEL_BOUND/X_BOUND;
        double zi_fmod = fmod(zi, 500.0);
        if(fabs(zi) <= 500.0){
            sum += zi*sin(pow(fabs(zi),0.5));
        } else if(zi > 500.0) {
            sum += (500 - zi_fmod)*sin(pow(fabs(500 - zi_fmod), 0.5)) - (zi - 500)*(zi - 500)/(10000*ceil(nx*HF06_n4));
        } else{
            sum += (-500.0+fmod(fabs(zi),500.0))*sin(pow(500.0-fmod(fabs(zi),500.0),0.5)) - (zi + 500)*(zi + 500)/(10000*ceil(nx*HF06_n4));
        }
    }

    smem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        fit += 4.189828872724338e+002 * ceil(nx*HF06_n4) - smem[gene_block_id];

    }

    // ACKLEY
    i = threadIdx.x + n_f;
    sum = 0.0;
    sum2 = 0.0;
    for( ; i < nx; i += blockDim.x){
        xi = x[i]*ACKLEY_BOUND/X_BOUND;

        sum += xi*xi;
        sum2 += cos(2*PI*xi);
    }

    smem[gene_block_id] = sum;
    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        sum = smem[gene_block_id];
    }
    
    smem[gene_block_id] = sum2;
    __syncthreads();
    reduction(gene_block_id, smem);

    if(threadIdx.x == 0){
        sum2 = smem[gene_block_id];
        double c = 1.0 / double(nx-n_f);

        f[chromo_id]  = fit + 20 - 20*exp(-0.2*sqrt( c * sum)) + E - exp(c * sum2);  

    }
}

__global__ void cf_cal_gpu(double *x, double *f, double *Os, double *lambda, double *delta, double *bias, double *fit, int nx){
    // cf_index = threadIdx.y
    
    int i;
    cg::thread_block_tile<WARP_SIZE> group = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

    __shared__ double smem[WARP_SIZE];

    double xi;
    double w = 0.0;
    double f_temp = 0.0;        

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

    double w_sum = 0.0;
    double w_temp = 0.0;

    if(group.meta_group_rank() == 0){
        w = 0;
        if(group.thread_rank() < blockDim.y){
            w = smem[group.thread_rank()];
            double d = delta[group.thread_rank()];
            double b = bias[group.thread_rank()];
            
            f_temp = fit[group.thread_rank()]*lambda[group.thread_rank()] + b;
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