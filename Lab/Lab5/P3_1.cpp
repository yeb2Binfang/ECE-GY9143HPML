#include<math.h>
#include<iostream>
#include <time.h>

//function to add the elements of two arrays
void add(int n, float *x, float *y){
        for(int i = 0; i < n; i++){
                y[i] = x[i] + y[i];
        }
}

int main(void){
	int N = 50<<20;

	float *x = new float[N];
	float *y = new float[N];

	for(int i = 0; i < N; i++){
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
	//time_t begin, end;
	//time(&begin);
	add(N, x, y);
	std::cout << "Done" << std::endl;
	//time(&end);
	//time_t elapsed = end - begin;
	//printf("Time measured: %.3f seconds.\n", elapsed);
	
	

	//float maxError = 0.0f;
	//for(int i = 0; i < N; i++){
	//	maxError = fmax(maxError, fabs(y[i] - 3.0f));
	//}

	//std::cout << "Max Error: " << maxError << std::endl;

	//free memory
	delete [] x;
	delete [] y;

	return 0;
}
