#include "Data.h"
#include "../../utils/utils.h"
#include <stdio.h>
#include <stdlib.h>

//DIM defined at compile time by -DDIM=X
#ifndef DIM
    #define DIM 10
#endif

#ifndef MXOUT
    #define MXOUT ./matrix.bin
#endif

#ifndef TYPE_M
    #define TYPE_M float 
#endif

#ifndef SOUT
    #define SOUT ./shift_vec.bin
#endif

#ifndef BOUNDARY
    #define BOUNDARY 80.0
#endif

#define QUOTE(s) #s
#define STRING(macro) QUOTE(macro)

#define MXOUTNAME STRING(MXOUT)
#define SOUTNAME  STRING(SOUT)

int main(int argc, char *argv[]){
#if IS_MX == 1
    SquareMatrixData<TYPE_M, DIM> data;
    data.rotm_init();
    data.to_binary(MXOUTNAME);
#elif IS_MX == 0
    VectorData<TYPE_M, DIM> data;
    data.random_init(-BOUNDARY, BOUNDARY);
    data.to_binary(SOUTNAME);
#else
    SquareMatrixData<TYPE_M, DIM> data;

    TYPE_M *d = (TYPE_M*)malloc(8*DIM*DIM*sizeof(TYPE_M));
    FILE *fp;

    fp = fopen(MXOUTNAME, "w+");

    for(int i = 0; i < 8; i++){
        data.rotm_init();
        d = data.data();

        fwrite(d, DIM*sizeof(TYPE_M), DIM, fp);

    }

    fclose(fp);

#endif


    return 0;
}