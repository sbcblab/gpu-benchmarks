#!/bin/bash

type=$1
dim=$2

cd input_data

./composition_matrix.sh $type $dim composition_$type_$dim.bin

./rot_matrix.sh $type $dim basic_$1_$dim.bin

./shift_vector.sh $type $dim 80 basic_$type_$dim.bin

./shift_vector.sh $type $(($dim * 8)) 80 composition_$type_$dim.bin

./shuffle_vector.sh $dim shuffle_$dim.bin