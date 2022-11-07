# Random Rotation Matrix Generation
This script creates a square NxN matrix using the Gram-Schmidt orthonormalization process. The generated matrices are stored into the `./matrices` directory, which is created during the execution.

## Configuring

The script needs permissions to execute
```
chmod +x ./rot_matrix.sh

```

## Execution

```
    ./rot_matrix.sh TYPE DIM_MATRIX FILENAME
```

For instance, the command `./rot_matrix.sh double 10 matrix10x10.bin` will create a random 10x10 rotation matrix and store it into `./matrices/matrix10x10.bin` 

# Random Shift Vector Generation
This script creates a random shift vector of lenght _n_. The generated vectors are stored into the `./shift_vectors` directory, which is created on the first execution.

## Configuring

The script needs permissions to execute
```
chmod +x ./shift_vector.sh

```

## Execution

```
    ./shift_vector.sh TYPE DIM_VECTOR BOUNDARY FILENAME
```

For instance, the command `./shift_vector.sh double 10 80 shift.bin` will create a random vector of length 10 with values whithin the range [-80, 80]. It stores the vector into the file `./shift_vectors/shift.bin` 
