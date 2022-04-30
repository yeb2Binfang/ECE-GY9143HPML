#include<cassert>
#include<cstdlib>
#include<iostream>

// 3 x 3 x 3 convolutional mask
#define MASK_DIM 3


//2d convolution
//takes:
//	matrix: input matrix
//	result: the convolution result
//	N: dimensions of the matrices
__global__ void convolution_2d(int color, int rows, int cols, int kRows, int kCols, int ***matrix, int ***kernel, int ***result){
 int kCenterX = kCols / 2;
 int kCenterY = kRows / 2;
	
  for(int c = 0; c < color;c++){
    for(int i=0; i < rows; ++i) {// rows
      for(int j=0; j < cols; ++j){// columns
        for(int m=0; m < kRows; ++m) {// row index of flipped kernel
          int mm = kRows - 1 - m; // row index of flipped kernel
          for(int n=0; n < kCols; ++n){// kernel columns
             int nn = kCols - 1 - n;  // column index of flipped kernel
             // index of input signal, used for checking boundary
             int ii = i + (kCenterY - mm);
             int jj = j + (kCenterX - nn);
              // ignore input samples which are out of bound
             if( ii >= 0 && ii < rows && jj >= 0 && jj < cols ){
                result[c][i][j] += matrix[c][ii][jj] * kernel[c][mm][nn];
             }
          }
        }
      }
    }             
}

//init the n x n matrix
//m: pointer to the matrix
//n: dimension of the matrix
void init_matirx(int ***m, int N, int C){
	for(int c = 0;c < C;c++){
		for(int i = 0;i < N;i++){
			for(int j = 0;j < N; j++){
				m[c][i][j] = rand() % 100;
			}
		}
	}
	
}

void verify_result(int color, int rows, int cols, int kRows, int kCols, int ***matrix, int ***kernel, int ***result){
	int temp;
	int check_sum = 0;
	int kCenterX = kCols / 2;
  	int kCenterY = kRows / 2;
	
  for(int c = 0; c < color;c++){
    for(int i=0; i < rows; ++i) {// rows
      for(int j=0; j < cols; ++j){// columns
        temp = 0;
        for(int m=0; m < kRows; ++m) {// row index of flipped kernel
          int mm = kRows - 1 - m; // row index of flipped kernel
          for(int n=0; n < kCols; ++n){// kernel columns
             int nn = kCols - 1 - n;  // column index of flipped kernel
             // index of input signal, used for checking boundary
             int ii = i + (kCenterY - mm);
             int jj = j + (kCenterX - nn);
              // ignore input samples which are out of bound
             if( ii >= 0 && ii < rows && jj >= 0 && jj < cols ){
                 += matrix[c][ii][jj] * kernel[c][mm][nn];
             }
          }
        }
        check_sum += result[c][i][j] - temp; 
      }
    }             
	
    std::cout << "The check sum is "<< check_sum;
}

int main(){
	int N = 1024;
	int C = 3;
	size_t bytes_n = N * N * C * sizeof(int);

	int ***matrix = new int[C][N][N];
	int ***result = new int[C][N][N];

	//init the matrix
	init_matirx(matrix, N, C);

	size_t bytes_m = MASK_DIM * MASK_DIM * C * sizeof(int);

	int ***h_mask = new int[C][MASK_DIM][MASK_DIM];
	init_matirx(h_mask, MASK_DIM, C);

	int ***d_matrix;
	int ***d_result;

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
