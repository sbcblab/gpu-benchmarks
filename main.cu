#include <stdio.h>
#include "functions/Zakharov.cuh"
#include "functions/Rosenbrock.cuh"
#include "functions/Levy.cuh"
#include "functions/SchafferF6.cuh"
#include "functions/StepRastrigin.cuh"
#include "functions/Hybrid01.cuh"
#include "functions/Hybrid02.cuh"
#include "functions/Hybrid03.cuh"
#include "functions/Composition01.cuh"
#include "functions/Composition02.cuh"
#include "functions/Composition03.cuh"
#include "functions/Composition04.cuh"
#include "utils.h"

void run_function(double *x_d, int n, int pop, int func_num, int ipb);
// void run_cpu_benchmark(double *x, int n, int pop, int func_num);

int main(int argc, char *argv[]){
    int ipb = read_ipb(argc, argv);
    int n = read_dimensions(argc, argv);
    int pop = read_population(argc, argv);
    int func_num = read_function(argc, argv);
    

    double *x  = (double*)malloc(sizeof(double)*n*pop);
    double *x_d;

    cudaMalloc<double>(&x_d, sizeof(double)*n*pop);
    // run_cpu_benchmark(x, n, pop, func_num);

    for(int i =0; i < 20; i++){
        init_random_vector<double>(x, n*pop, -100.0, 100.0);
        
        cudaMemcpy(x_d, x, sizeof(double)*n*pop, cudaMemcpyHostToDevice);

        run_function(x_d, n, pop, func_num, ipb);    
    }

    free(x);
    cudaFree(x_d);
    return 0;
}
/*
void run_cpu_benchmark(double *x, int n, int pop, int func_num){
    init_benchmark_cpu(n);
    double *f = (double *)malloc(sizeof(double)*pop);
    double *Mr;
    double *Os;
    char matrix_filename[50] = {};

    if(func_num < F_COMPOSITION1){

        Mr = (double *)malloc(sizeof(double)*n*n);
        Os = (double *)malloc(sizeof(double)*n);
        
        snprintf(matrix_filename, 50, "../input_data/matrices/basic_%d.bin", n);

        init_vector_from_binary<double>(Mr, n*n, matrix_filename);
        init_vector_from_binary<double>(Os, n, "../input_data/shift_vectors/basic_shift.bin");
    } else {
        int cf_num;
        switch(func_num){
            case F_COMPOSITION2:
                cf_num = 3;
                break;
            case F_COMPOSITION4:
                cf_num = 6;
                break;
            case F_COMPOSITION1:
            case F_COMPOSITION3:
                cf_num = 5;
                break;
        }
        Mr = (double *)malloc(sizeof(double)*cf_num*n*n);
        Os = (double *)malloc(sizeof(double)*cf_num*n);
        
        snprintf(matrix_filename, 50, "../input_data/matrices/composition_%d.bin", n);

        init_vector_from_binary<double>(Mr, cf_num*n*n, matrix_filename);
        init_vector_from_binary<double>(Os, cf_num*n, "../input_data/shift_vectors/composition_shift.bin");
    }

    for(int i = 0; i < pop; i++){
        switch(func_num){
            case F_ZAKHAROV: {
                zakharov_func(&x[i*n], &f[i], n, Os, Mr, 1, 1);
                break;
            }
            case F_ROSENBROCK: {
                rosenbrock_func(&x[i*n], &f[i], n, Os, Mr, 1, 1);

                break;
            }
            case F_LEVY: {
                levy_func(&x[i*n], &f[i], n, Os, Mr, 1, 1);
                break;
            }
            case F_SCHAFFER_F6:{
                escaffer6_func(&x[i*n], &f[i], n, Os, Mr, 1, 1);

                break;
            }
            case F_STEP_RASTRIGIN:{
                rastrigin_func(&x[i*n], &f[i], n, Os, Mr, 1, 1);

                break;
            }
            case F_HYBRID1: {

                hf02_func(&x[i*n], &f[i], n, Os, Mr, NULL, 1, 1);

                break;
            }
            case F_HYBRID2: {
                hf06_func(&x[i*n], &f[i], n, Os, Mr, NULL, 1, 1);

                break;
            }
            case F_HYBRID3: {
                hf10_func(&x[i*n], &f[i], n, Os, Mr, NULL, 1, 1);

                break;
            }
            case F_COMPOSITION1: {
                cf01(&x[i*n], &f[i], n, Os, Mr, 1);

                break;
            }
            case F_COMPOSITION2: {
                cf02(&x[i*n], &f[i], n, Os, Mr, 1);

                break;
            }
            case F_COMPOSITION3: {
                cf06(&x[i*n], &f[i], n, Os, Mr, 1);
                
                break;
            }
            case F_COMPOSITION4: {
                cf07(&x[i*n], &f[i], n, Os, Mr, 1);
                break;
            }
        }

    }
    
    print_vector<double>(f, pop);    
 
    free_benchmark_cpu();
    free(f);
    free(Os);
    free(Mr);
}
*/

void run_function(double *x_d, int n, int pop, int func_num, int ipb){

    double *f  = (double*)malloc(sizeof(double)*pop);
    double *f_d;
    
    cudaMalloc<double>(&f_d, sizeof(double)*pop);

    dim3 evaluation_block(min(n/ipb, 1024/ipb), ipb);
    int evaluation_grid = pop/ipb + pop % ipb ;

    switch(func_num){
        case F_ZAKHAROV: {
            Zakharov<double> bench(n, pop);
            bench.set_launch_config(evaluation_grid, evaluation_block);

            bench.compute(x_d, f_d);

            break;
        }
        case F_ROSENBROCK: {
            Rosenbrock<double> bench(n, pop);
            bench.set_launch_config(evaluation_grid, evaluation_block);

            bench.compute(x_d, f_d);

            break;
        }
        case F_LEVY: {
            Levy<double> bench(n, pop);
            bench.set_launch_config(evaluation_grid, evaluation_block);

            bench.compute(x_d, f_d);

            break;
        }
        case F_SCHAFFER_F6:{
            SchafferF6<double> bench(n, pop);
            bench.set_launch_config(evaluation_grid, evaluation_block);

            bench.compute(x_d, f_d);

            break;
        }
        case F_STEP_RASTRIGIN:{
            StepRastrigin<double> bench(n, pop);
            bench.set_launch_config(evaluation_grid, evaluation_block);

            bench.compute(x_d, f_d);
            break;
        }
        case F_HYBRID1: {
            Hybrid01<double> bench(n, pop);
            bench.set_launch_config(evaluation_grid, evaluation_block);

            bench.compute(x_d, f_d);
            break;
        }
        case F_HYBRID2: {
            Hybrid02<double> bench(n, pop);
            bench.set_launch_config(evaluation_grid, evaluation_block);

            bench.compute(x_d, f_d);

            break;
        }
        case F_HYBRID3: {
            Hybrid03<double> bench(n, pop);
            bench.set_launch_config(evaluation_grid, evaluation_block);


            bench.compute(x_d, f_d);
            break;
        }
        case F_COMPOSITION1: {
            Composition01<double> bench(n, pop);
            bench.set_launch_config(evaluation_grid, evaluation_block);

            bench.compute(x_d, f_d);
            break;
        }
        case F_COMPOSITION2: {
            Composition02<double> bench(n, pop);
            bench.set_launch_config(evaluation_grid, evaluation_block);

            bench.compute(x_d, f_d);
            break;
        }
        case F_COMPOSITION3: {
            Composition03<double> bench(n, pop);
            bench.set_launch_config(evaluation_grid, evaluation_block);

            bench.compute(x_d, f_d);
            break;
        }
        case F_COMPOSITION4: {
            Composition04<double> bench(n, pop);
            bench.set_launch_config(evaluation_grid, evaluation_block);

            bench.compute(x_d, f_d);

            break;
        }
    }

    cudaMemcpy(f, f_d, sizeof(double)*pop, cudaMemcpyDeviceToHost);
    
    print_vector<double>(f, pop);

    cudaFree(f_d);
    free(f);
}