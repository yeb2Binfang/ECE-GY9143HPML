#include <stdio.h>
#include <stdlib.h>
#include <time.h>
struct timespec start, end;

float dpunroll(long N, float *pA, float *pB){
    float R = 0.0;
    int j;
    for(j=0;j<N;j += 4){
        R += pA[j] * pB[j] + pA[j+1] * pB[j+1] + pA[j+2] * pB[j+2] + pA[j+3] * pB[j+3];
    }
    return R;
}

int main(int argc, char* argv[]){
    printf("start to measure\n");
    double avg_time = 0.0;
    long N = atoi(argv[1]);
    int times = atoi(argv[2]);
 //   printf("N = %d", N);
  //  printf("times = %d", times);
    float *pA = malloc(N * sizeof(float));
    float *pB = malloc(N * sizeof(float));
    //float pA[N];
    //float pB[N];
    for(int i=0;i<N;i++){
   	   pA[i] = 1.0;
  	   pB[i] = 1.0;
    }

    float R = 0;
    for(int i=0;i<times;i++){
       clock_gettime(CLOCK_MONOTONIC, &start);
       R = dpunroll(N, pA, pB);
       clock_gettime(CLOCK_MONOTONIC, &end);
       if(i >= (times / 2 - 1)){
       	 double timespec = (((double)end.tv_sec * 1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec * 1000000 + (double)start.tv_nsec/1000));
	
         avg_time += timespec;
       }
    
    }
    avg_time = avg_time / times / 2;
    double bandwidth = 1000000 * (double)N * sizeof(float) * 2 / avg_time / 1000000000;
    double flops = 1000000 * 8 * (double)N / avg_time / 1000000000;
    printf("N:%ld ,time: %.07lfs, B:%.3lf GB/s, GFLOPS:%.3lf ",N, avg_time / 1000000, bandwidth, flops);
    free(pA);
    free(pB);
    return 0;
}



