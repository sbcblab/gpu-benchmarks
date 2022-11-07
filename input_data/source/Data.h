#pragma once

#include <cstdlib>
#include <stdio.h>
#include <random>
#include <math.h>
#include <array>
#include "RotationMatrix.h"

template<typename T, size_t size_array>
class VectorData{
	private:
		std::array<T, size_array> d;
	public:
		VectorData(){}
		virtual ~VectorData(){}
		
		T* data(){
			return d.data();
		}

		void random_init(int min, int max){
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<T> dis(min, max);
			for(int i=0; i < d.size(); i++){
				d[i] = dis(gen);
			}
		}

		void to_binary(const char*filename){
			FILE *fp;
			fp = fopen(filename, "wb");

			if(!fp){
				perror("Error opening binary file");
				exit(EXIT_FAILURE);
			}

			fwrite(d.data(), sizeof(T), d.size(), fp);
			fclose(fp);
		}

		void from_binary(const char*filename){
			FILE *fp;

			fp = fopen(filename, "rb");
			
			if(!fp){
				perror("Error opening binary file");
				exit(EXIT_FAILURE);
			}
			
			fread(d.data(), sizeof(T), d.size(), fp);
			fclose(fp);
		}

		void print(){
			int n = d.size();
			printf("[ ");
    
			if(std::is_same<T, double>::value){
				for(int i = 0; i < n-1; i++){
					printf("%lf, ", d[i]);   
				}

				printf("%lf ]\n", d[n-1]);   
			} 
			else if(std::is_same<T, float>::value){
				for(int i = 0; i < n-1; i++){
					printf("%f, ", d[i]);   
				}

				printf("%f ]\n", d[n-1]);   
			} 
		}

};

template<typename T, size_t size_array>
class SquareMatrixData{
	private:
		std::array<T, size_array*size_array> d;
		int n;

	public:
		SquareMatrixData(){
			n = size_array;
		}
		virtual ~SquareMatrixData(){}

		T* data(){
			return d.data();
		}


		void rotm_init(){
			rotation_matrix(d.data(), n);
		}

		void random_init(int min, int max){
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<T> dis(min, max);
			for(int i=0; i < d.size(); i++){
				d[i] = dis(gen);
			}
		}

		void to_binary(const char*filename){
			FILE *fp;
			fp = fopen(filename, "wb");

			if(!fp){
				perror("Error opening binary file");
				exit(EXIT_FAILURE);
			}

			fwrite(d.data(), sizeof(T), d.size(), fp);
			fclose(fp);
		}

		void from_binary(const char*filename){
			FILE *fp;

			fp = fopen(filename, "rb");
			
			if(!fp){
				perror("Error opening binary file");
				exit(EXIT_FAILURE);
			}
			
			fread(d.data(), sizeof(T), d.size(), fp);
			fclose(fp);
		}

		void print(){
			int n = (int)sqrt(d.size());
			for(int i = 0; i < n; i++){
				if(std::is_same<T, double>::value){
					for(int j = 0; j < n-1; j++){
						printf("%lf ", d[i*n + j]);
					}

					printf("%lf\n", d[i*n + n-1]);    
				}
				else if(std::is_same<T, float>::value){
					for(int j = 0; j < n-1; j++){
						printf("%f ", d[i*n + j]);
					}

					printf("%f\n", d[i*n + n-1]);  
				}
			}
		}

};

