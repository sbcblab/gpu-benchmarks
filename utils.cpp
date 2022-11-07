#include <math.h>
#include <string.h>
#include "utils.h"

#define FLAG_NOT_FOUND  -1

#define POP_SIZE_DEFAULT    8 
#define DIMENSIONS_DEFAULT  8
#define FUNC_DEFAULT        0

// return index of flag
static int find_flag(const char *flag, int argc, char *argv[]){
    int i; 
    
    for(i = 1; i < argc; i += 1){
        if(strcmp(argv[i], flag) == 0){
            return i;
        }
    }
    return FLAG_NOT_FOUND;
}

int next_pows(int n){
    return pow(2, ceil(log2((double)n)));
}

int read_ipb(int argc, char *argv[]){
    int ipb = 1;
    
    int i_flag = find_flag("-i", argc, argv);

    if(i_flag == FLAG_NOT_FOUND){
        return ipb;
    } else {
        if( i_flag < argc ){
            ipb = atoi(argv[i_flag + 1]);
            return ipb;
        } else {
            printf("No arguments for -i flag\n");
            exit(1);
        }
    }
}

int read_dimensions(int argc, char *argv[]){
    int dim = DIMENSIONS_DEFAULT;
    
    int i_flag = find_flag("-d", argc, argv);

    if(i_flag == FLAG_NOT_FOUND){
        return dim;
    } else {
        if( i_flag < argc ){
            dim = atoi(argv[i_flag + 1]);
            return dim;
        } else {
            printf("No arguments for -d flag\n");
            exit(1);
        }
    }
}

int read_population(int argc, char *argv[]){
    int pop = POP_SIZE_DEFAULT;
    
    int i_flag = find_flag("-p", argc, argv);

    if(i_flag == FLAG_NOT_FOUND){
        return pop;
    } else {
        if( i_flag + 1 < argc){
            pop = atoi(argv[i_flag+1]);
            return pop;
        } else {
            printf("No arguments for -p flag\n");
            exit(1);
        }
    }

}

int read_function(int argc, char *argv[]){
    int func = FUNC_DEFAULT;
    
    int i_flag = find_flag("-f", argc, argv);

    if(i_flag == FLAG_NOT_FOUND){
        return func;
    } else {
        if( i_flag + 1 < argc){
            func = atoi(argv[i_flag+1]);
            return func;
        } else {
            printf("No arguments for -p flag\n");
            exit(1);
        }
    }

}

int read_shift(int argc, char *argv[]){
    int i_flag = find_flag("-shf", argc, argv);

    if(i_flag == FLAG_NOT_FOUND){
        return 0;
    } else {
        return 1;
    }
}

int read_rotate(int argc, char *argv[]){
    int i_flag = find_flag("-rot", argc, argv);

    if(i_flag == FLAG_NOT_FOUND){
        return 0;
    } else {
        return 1;
    }
}



void print_time(clock_t t, const char msg[]){
	printf("%s: %lf\n", msg, 1000*((double)t)/CLOCKS_PER_SEC);
}


void mul_matrix_vector_cpu(double *M, double *V, double *out, int n){
    int i,j;
    for (i=0; i<n; i++){
        out[i]=0;
		for (j=0; j<n; j++){
			out[i] += V[j]*M[i*n+j];
		}
    }
}