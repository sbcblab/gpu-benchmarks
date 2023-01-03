#include "Data.h"
#include "../utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <random>

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

#ifndef SHUFFLEOUT
    #define SHUFFLEOUT ./shuffle.bin
#endif

#ifndef BOUNDARY
    #define BOUNDARY 80.0
#endif

#define QUOTE(s) #s
#define STRING(macro) QUOTE(macro)

#define MXOUTNAME STRING(MXOUT)
#define SOUTNAME  STRING(SOUT)
#define SHUFFLEOUT_NAME STRING(SHUFFLEOUT)

int main(int argc, char *argv[]){

#if IS_MX == 1
    SquareMatrixData<TYPE_M, DIM> data;
    data.rotm_init();
    data.to_binary(MXOUTNAME);

#elif IS_MX == 0
    VectorData<TYPE_M, DIM> data;
    data.random_init(-BOUNDARY, BOUNDARY);
    data.to_binary(SOUTNAME);

#elif IS_MX == 2
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

#elif IS_MX == 3
    std::vector<int> indices;

    for(int i = 0; i < DIM; i++){
        indices.push_back(i);
    }

    // shuffle indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(std::begin(indices), std::end(indices), rd);

    printf("%s\n", SHUFFLEOUT_NAME);

    // write vector to file
    FILE *fp = fopen(SHUFFLEOUT_NAME, "w+");
    fwrite(indices.data(), sizeof(int), DIM, fp);
    fclose(fp);

#endif


    return 0;
}