#include <iostream>
#include <math.h>

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__ void add(int *x, int *y, int n){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	
	y[tid] = x[tid] + y[tid];
}

int main(){
	int n = 100<<20;

	int bytes_n = n * sizeof(int);

	int *x = new int[n];
	int *y = new int[n];

	for(int i = 0; i < n; i++){
		x[i] = 1;
		y[i] = 2;
	}
	int *x_array, *y_array;


	cudaMalloc(&x_array, bytes_n);
	cudaMalloc(&y_array, bytes_n);

	cudaMemcpy(x_array, x, bytes_n, cudaMemcpyHostToDevice);
	cudaMemcpy(y_array, y, bytes_n, cudaMemcpyHostToDevice);

	int THREADS = 256;

	int GRID = (n + THREADS - 1) / THREADS;;

	add<<<GRID, THREADS>>>(x_array, y_array, n);
	cudaMemcpy(y, y_array, bytes_n, cudaMemcpyDeviceToHost);

	delete[] x;
	delete[] y;
	cudaFree(x_array);
	cudaFree(y_array);
	return 0;
}
