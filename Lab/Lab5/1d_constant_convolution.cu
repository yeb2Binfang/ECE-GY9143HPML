#include<cassert>
#include<cstdlib>
#include<iostream>

#define MASK_LENGTH 7

__constant__ int mask[MASK_LENGTH]

__global__ void convolution_1d(int *array, int *result, int n){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int r = MASK_LENGTH / 2;
	
	int start = tid - r;

	int temp = 0;

	for(int j = 0; j < MASK_LENGTH; j++){
		if(start + j >= 0 && start + j < n){
			temp += array[start + j] * mask[j];
		}
	}

	result[tid] = temp;
}

void verify_result(int *array, int *mask, int *result, int n){
	int radius = MASK_LENGTH / 2;
	int temp;
	int start;
	for(int i = 0; i < n; i++){
		start = i - radius;
		temp = 0;
		for(int j = 0; j < MASK_LENGTH; j++){
			if(start + j >= 0 && start + j < n){
				temp += array[start + j] * mask[j];
			}	
		}
		assert(temp == result[i]);
		
	}
}

int main(){
	int n = 1 << 20;

	int bytes_n = n * sizeof(int);

	size_t bytes_m = MASK_LENGTH * sizeof(int);

	int *h_array = new int[n];

	for(int i = 0; i < n; i++){
		h_array[i] = rand()  % 100;
	}

	int *h_result = new int[n];

	int *d_array, *d_result;

	cudaMalloc(&d_array, bytes_n);
	cudaMalloc(&d_result, bytes_n);

	cudaMemcpy(d_array, h_array, bytes_n, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(mask, h_mask, bytes_m);

	int THREADS = 256;

	int GRID = (n + THREADS - 1) / THREADS;

	convolution_1d<<<<GRID, THREADS>>>>(d_array, d_result, n);

	cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

	verify_result(h_array, h_mask, h_result, n);

	std::cout << "compiled successfully\n";

	delete[] h_array;
	delete[] h_result;
	delete[] h_mask;
	cudaFree(d_result);

	return 0;

}
