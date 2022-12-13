#!/bin/bash
cd input_data

./composition_matrix.sh $1 $2 composition_$1_$2.bin

./rot_matrix.sh $1 $2 basic_$1_$2.bin

./shift_vector.sh $1 $2 80 shift_$1_$2.bin

./shuffle_vector.sh $2 shuffle_$2.bin