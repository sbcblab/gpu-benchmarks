#!/bin/bash
DIR=./matrices
mkdir -p $DIR

if [ $# -eq 0 ] 
    then 
        make composition dim=10 mxout=$DIR/matrix.bin
else
    make composition type=$1 dim=$2 mxout=$DIR/$3
fi

./matrix.out 