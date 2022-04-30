#include<cassert>
#include<cstdlib>
#include<iostream>

// 3 x 3 convolutional mask
#define MASK_DIM 3

#define MASK_OFFSET (MASK_DIM / 2)

__constant__ int mask[3 * 3];

//2d convolution
//takes:
//	matrix: input matrix
//	result: the convolution result
//	N: dimensions of the matrices
__global__ void convolution_2d(int **matrix, int **result, int N, int C){
		
	for(int c = 0;c < C;c++){
		int temp = 0;
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;

		int start_r = row - MASK_OFFSET;
		int start_c = col - MASK_OFFSET;
		for(int i = 0;i < MASK_DIM;i++){
			for(int j = 0;j < MASK_DIM; j++){
				if(start_r + i >= 0 && start_r + i < N){
					if(start_c + j >= 0 && start_c + j < N){
						temp += matrix[(start_r + i) * N + (start_c + j)] * mask[i * MASK_DIM + j];
					}
				}
			}
		}
		result[c][row * N + col] = temp;
	}
	
	
}

//init the n x n matrix
//m: pointer to the matrix
//n: dimension of the matrix
void init_matirx(int **m, int n, int c){
	for(int k = 0;k < c;k++){
		for(int i = 0;i < n;i++){
			for(int j = 0;j < n; j++){
				m[k][n * i + j] = rand() % 100;
			}
		}
	}
	
}

void verify_result(int **m, int **mask, int **result, int N, int C){
	int temp;
	int check_sum = 0;
	
	int offset_r;
	int offset_c;
	for(int c = 0;c < C;c++){
		for(int i = 0;i < N;i++){
			for(int j = 0;j < N;j++){
				temp = 0;

				for(int k = 0;k < MASK_DIM;k++){
					offset_r = i - MASK_OFFSET + k;
					for(int l = 0;l < MASK_DIM;l++){
						offset_c = j - MASK_OFFSET + l;

						if(offset_r >= 0 && offset_r < N){
							if(offset_c >= 0 && offset_c < N){
								temp += m[offset_r * N + offset_c] * mask[k * MASK_DIM + l];
							}
						}
					}

				}
				check_sum += result[i * N + j] - temp;
			}
		
		}
	}
	
	std::cout << "The check sum is "<< check_sum;
}

int main(){
	int N = 1024;
	int C = 3;
	size_t bytes_n = N * N * C * sizeof(int);

	int **matrix = new int[3][N * N];
	int **result = new int[3][N * N];

	//init the matrix
	init_matirx(matrix, N, C);

	size_t bytes_m = MASK_DIM * MASK_DIM * C * sizeof(int);

	int **h_mask = new int[C][MASK_DIM * MASK_DIM];
	init_matirx(h_mask, MASK_DIM);

	int **d_matrix;
	int **d_result;

	cudaMalloc(&d_matrix, bytes_n);
	cudaMalloc(&d_result, bytes_n);

	cudaMemcpy(d_matrix, matrix, bytes_n, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(mask, h_mask, bytes_m);

	int THREADS = 32;
	int BLOCKS = (N + THREADS - 1) / THREADS;

	dim3 block_dim(THREADS, THREADS);
	dim3 grid_dim(BLOCKS, BLOCKS);
	
	convolution_2d<<<grid_dim, block_dim>>>(d_matrix, d_result, N, C);

	cudaMemcpy(result, d_result, bytes_n, cudaMemcpyDeviceToHost);

	verify_result(matrix, h_mask, result, N);

	std::cout <<"Completed successfully!";

	delete[] matrix;
	delete[] result;
	delete[] h_mask;

	cudaFree(d_matrix);
	cudaFree(d_result);
	
	return 0;

}
