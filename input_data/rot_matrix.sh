#!/bin/bash
DIR=./matrices
mkdir -p $DIR

if [ $# -eq 0 ] 
    then 
        make dim=10 mxout=$DIR/matrix.bin
else
    make type=$1 dim=$2 mxout=$DIR/$3
fi

./matrix.out 
