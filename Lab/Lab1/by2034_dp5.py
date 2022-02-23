'''

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

'''
import sys
import numpy as np
import time

N = sys.argv[1]
times = sys.argv[2]

N = int(N)
times = int(times)

A = np.ones(N, dtype = np.float32)
B = np.ones(N, dtype = np.float32)

def dp(N, A, B):
    return np.dot(A.T, B);


avg_time = 0
for i in range(0, times):
  
  if i >= times / 2-1:
     start = time.time()
     R = dp(N, A, B)
     end = time.time()
     avg_time += (end -start)
     
  else:
     R = dp(N, A, B)
     
avg_time = avg_time / (times/2)
bandwidth = 2 * 4 * N / avg_time / 1000000000
flop = ((times / 2) * N * 2) / avg_time / 1000000000
print("N: {0}, time: {1}s, bandwidth: {2}GB/s, Gflops: {3}".format(N, avg_time, bandwidth, flop))
