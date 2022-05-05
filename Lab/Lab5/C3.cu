#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>

template<typename T>
__global__ void conv2d(int batch, int color, int rows, int cols, int kCols, int kRows, T* matrix, float* kernel, T* result){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
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
								result[b * color * rows * cols + c * rows * cols + i * cols + j] += matrix[b * c * ii * jj + c * ii * jj + ii * kRows + jj] * kernel[mm * kRows + nn];
								result[tid] = result[b * color * rows * cols + c * rows * cols + i * cols + j];
							}
						}
					}
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

void verify_result(float* matrix, float* kernel, float* result){
	int check_sum = 0;
	int kRows = 3;
	int kCols = 3;
	int rows = 1024;
	int cols = 1024;
	int color = 3;
	int kCenterX = kCols / 2;
        int kCenterY = kRows / 2;

        for(int b = 0; b < 64; b++){
                for(int c = 0; c < 3; c++){
                        for(int i = 0; i < rows;i++){
                                for(int j = 0; j < cols; j++){
                                        for(int m = 0; m <kRows; m++){
                                                int mm = kRows - 1 - m;
                                                for(int n = 0; n < kCols; n++){
                                                        int nn = kCols - 1 - n;

                                                        int ii = i + (kCenterY - mm);
                                                        int jj = j + (kCenterX - nn);

                                                        if(ii >= 0 && ii < rows && jj >= 0 && jj < cols){
                                                                result[b * color * rows * cols + c * rows * cols + i * cols + j] += matrix[b * c * ii * jj + c * ii * jj + ii * kRows + jj] * kernel[mm * kRows + nn];

                                                        }
                                                }
                                        }
                                }
                        }
                }
        }
	std::cout << "The check sum is" << check_sum << std::endl;
}

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

int main(){
	cudnnHandle_t cudnn;
 	cudnnCreate(&cudnn);
	cudnnTensorDescriptor_t input_descriptor;
  	checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	cudnnFilterDescriptor_t kernel_descriptor;
	checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
	cudnnConvolutionDescriptor_t convolution_descriptor;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
	int batch_size{0}, channels{0}, height{0}, width{0};
 	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   &batch_size,
                                                   &channels,
                                                   &height,
                                                   &width));
	cudnnTensorDescriptor_t output_descriptor;
  	checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
	cudnnConvolutionFwdAlgo_t convolution_algorithm;
 	checkCUDNN(
      	cudnnGetConvolutionForwardAlgorithm(cudnn,
                                          input_descriptor,
                                          kernel_descriptor,
                                          convolution_descriptor,
                                          output_descriptor,
                                          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                          /*memoryLimitInBytes=*/0,
                                          &convolution_algorithm));
		
	int n = 3 * 1024 * 1024;
	int size = 3 * 3 * 3;
	
	
	int bytes_n = n * sizeof(float);
	//int bytes_size = size * sizeof(float);

	float *matrix = new float[n];

	for(int i = 0; i < n; i++){
		int temp1 = i / 1024 / 1024; // channel
		int temp2 = i / 1024; //col
		int temp3 = i % 1024; //col
		matrix[i] = temp1 * (temp2 + temp3);
	}
	float *kernel = new float[size];
	for(int i = 0; i < size; i++){
		int temp1 = i / 3 / 3;
		int temp2 = i / 3;
		int temp3 = i % 3;
		kernel[i] = temp1 * (temp2 + temp3);
	}	

	float *result = new float[n];

	float *d_matrix, *d_result;
	cudaMalloc(&d_matrix, bytes_n);
  	cudaMalloc(&d_result, bytes_n);

	cudaMemcpy(d_matrix, matrix, bytes_n, cudaMemcpyHostToDevice);
	conv2d<<<1, 1>>>(64, 3, 1024, 1024, 3, 3, matrix, kernel, result);
	cudaDeviceSynchronize();
	verify_result(matrix, kernel, result);
	delete[] matrix;
 	delete[] kernel;
  	delete[] result;
	return 0;
}
