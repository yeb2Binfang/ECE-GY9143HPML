#include <stdio.h>
#include <stdlib.h>
#include <time.h>
struct timespec start, end;

float dp(long N, float *pA, float *pB){
    float R = 0.0;
    int j;
    for(j=0;j<N;j++){
        R += pA[j] * pB[j];
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
       R = dp(N, pA, pB);
       clock_gettime(CLOCK_MONOTONIC, &end);
       if(i >= (times / 2 - 1)){
       	 double timespec = (((double)end.tv_sec * 1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec * 1000000 + (double)start.tv_nsec/1000));
         avg_time += timespec;
       }
    
    }
    avg_time = avg_time / times / 2;
    double bandwidth = 1000000 * (double)N * sizeof(float) * 2 / avg_time / 1000000000;
    double flops = 1000000 * 2 * (double)N / avg_time / 1000000000;
    printf("Finished, N:%ld ,time: %.07lf s, B:%.3lf GB/s ,F:%.3lf GFLOPS",N, avg_time / 1000000, bandwidth, flops);
    free(pA);
    free(pB);
    return 0;
}



