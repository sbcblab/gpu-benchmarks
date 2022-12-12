#!/bin/bash
DIR=./shuffle_vectors
mkdir -p $DIR

if [ $# -eq 0 ] 
    then 
        make shuffle dim=10 sout=$DIR/shuffle.bin
else
    make shuffle dim=$1 sout=$DIR/$2 
fi

./shuffle.out 
make clean