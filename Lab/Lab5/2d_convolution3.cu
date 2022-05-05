#include <cassert>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>

template<typename T>
__global__ void conv2d(int batch, int color, int rows, int cols, int kCols, int kRows, T**** matrix, float**** kernel, T**** result){
	int kCenterX = kCols / 2;
	int kCenterY = kRows / 2;

	for(int b = 0; b < batch; b++){
		for(int c = 0; c < color; c++){
			for(int i = 0; i < rows;i++){
				for(int j = 0; j < cols; j++){
					for(int m = 0; m <kRows; m++){
						int mm = kRows - 1 - m;
						for(int n = 0; n < kCols; n++){
							int nn = kCols - 1 - n;

							int ii = i + (kCenterY - mm);
							int jj = j + (kCenterX - nn);

							if(ii >= 0 && ii < rows && jj >= 0 && jj < cols){
								result[0][b][i][j] += matrix[0][c][ii][jj] * kernel[b][c][mm][nn];
							}
						}
					}
				}
			}
		}
	}
}


template<typename T>
T**** create_4d_flat(int a, int b, int c, int d){
	T *base;
	cudaMallocManaged(&base, a * b * c * d * sizeof(T));
	//assert(err == cudaSuccess);

	T ****ary;
	cudaMallocManaged(&ary, (a + a * b + a * b * c) * sizeof(T*));
	//assert(err == cudaSuccess);
	for (int i = 0; i < a; i++){
		ary[i] =  (T ***)((ary + a) + i * b);
		for (int j = 0; j < b; j++){
			ary[i][j] = (T **)((ary + a + a * b) + i * b * c + j * c);
			for (int k = 0; k < c; k++){
				ary[i][j][k] = base + ((i * b + j) * c + k) * d;
			}
		}
	}
	return ary;
}

template<typename T>
void free_4d_flat(T**** ary){
    if (ary[0][0][0]){
    	cudaFree(ary[0][0][0]);
    }
    if (ary){
    	cudaFree(ary);
    }
}

template<typename T>
__global__ void fill_matrix(T**** data, int a, int b, int c, int d){
	
	for (int i = 0; i < a; i++){
		for (int j = 0; j < b; j++){
			for (int k = 0; k < c; k++){
				for (int l = 0; l < d; l++){
					data[i][j][k][l] = j * (k + l);
				}
			}
		}
	}
}

template<typename T>
__global__ void fill_kernel(T**** data, int a, int b, int c, int d){

        for (int i = 0; i < a; i++){
                for (int j = 0; j < b; j++){
                        for (int k = 0; k < c; k++){
                                for (int l = 0; l < d; l++){
                                        data[i][j][k][l] = (i + j) * (k + l);
                                }
                        }
                }
        }
}

void report_gpu_mem(){
	 size_t free, total;
	 cudaMemGetInfo(&free, &total);
	 std::cout << "Free = " << free << " Total = " << total <<std::endl;
}

void verify_result(float**** matrix, float**** kernel, float**** result){
	int kCenterX = 3 / 2;
        int kCenterY = 3 / 2;
	int checksum = 0;
        int kRows = 3;
	int kCols = 3;
	int rows = 1024;
	int cols = 1024;
	for(int b = 0; b < 64; b++){
                for(int c = 0; c < 3; c++){
                        for(int i = 0; i < 1024;i++){
                                for(int j = 0; j < 1024; j++){
					int temp = 0;
                                        for(int m = 0; m <3; m++){
                                                int mm = kRows - 1 - m;
                                                for(int n = 0; n < 3; n++){
                                                        int nn = kCols - 1 - n;

                                                        int ii = i + (kCenterY - mm);
                                                        int jj = j + (kCenterX - nn);

                                                        if(ii >= 0 && ii < rows && jj >= 0 && jj < cols){
                                                                temp += matrix[0][c][ii][jj] * kernel[b][c][mm][nn];
                                                        }
                                                }
                                        }
					checksum += result[0][b][i][j] - temp;
                                }
                        }
                }
        }
	std::cout << "The check sum is" << checksum << std::endl;

}
int main(){
	report_gpu_mem();
	
	float**** matrix;
	float**** result;
	float**** kernel;
	std::cout << "allocating..." << std::endl;
	matrix = create_4d_flat<float>(1, 3, 1024, 1024);
	result = create_4d_flat<float>(1, 64, 1024, 1024);
	kernel = create_4d_flat<float>(64, 3, 3, 3);
	fill_matrix<<<1, 1>>>(matrix, 1, 3, 1024, 1024);
	fill_kernel<<<1, 1>>>(kernel, 64, 3, 3, 3);
	conv2d<<<1, 1>>>(64, 3, 1024, 1024, 3, 3, matrix, kernel, result);
	cudaDeviceSynchronize();
	//cudaError_t err = cudaDeviceSynchronize();
	//std::cout << data2[0][0][0][0] << std::endl;
	verify_result(matrix, kernel, result);
	free_4d_flat(matrix);
	free_4d_flat(kernel);
	free_4d_flat(result);
	return 0;
}
