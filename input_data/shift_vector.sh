#!/bin/bash
DIR=./shift_vectors
mkdir -p $DIR

if [ $# -eq 0 ] 
    then 
        make shift_vec dim=10 sout=$DIR/shift_vec.bin
else
    make shift_vec type=$1 dim=$2 bound=$3 sout=$DIR/$4 
fi

./shift_vec.out 
make clean