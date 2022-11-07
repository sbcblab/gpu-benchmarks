CC=g++
SRC=./source
OUT_FILE=matrix.out
OUT_VECTOR=shift_vec.out
UTILS=../utils

all: main.o


main.o: utils.o
	$(CC) -o $(OUT_FILE) -DDIM=$(dim) -DMXOUT=$(mxout) -DIS_MX=1 -DTYPE_M=$(type) $(SRC)/RotationMatrix.cpp  $(SRC)/main.cpp utils.o

shift_vec: utils.o
	$(CC) -o $(OUT_VECTOR) -DDIM=$(dim) -DSOUT=$(sout) -DIS_MX=0 -DBOUNDARY=$(bound) -DTYPE_M=$(type) $(SRC)/RotationMatrix.cpp  $(SRC)/main.cpp utils.o

composition: utils.o
	$(CC) -o $(OUT_FILE) -DDIM=$(dim) -DMXOUT=$(mxout) -DIS_MX=2 -DTYPE_M=$(type) $(SRC)/RotationMatrix.cpp  $(SRC)/main.cpp utils.o

utils.o: 
	$(CC) -c $(UTILS)/utils.cpp

clean:
	rm *.o *.out

