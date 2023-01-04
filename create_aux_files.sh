#!/bin/bash

type=$1
dim=$2

cd input_data

./composition_matrix.sh $type $dim composition_$1_$2.bin

./rot_matrix.sh $type $dim basic_$1_$2.bin

./shift_vector.sh $type $dim 80 basic_$1_$2.bin

./shift_vector.sh $type $(($dim * 8)) 80 composition_$1_$2.bin

./shuffle_vector.sh $dim shuffle_$2.bin