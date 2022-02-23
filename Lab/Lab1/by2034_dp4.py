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
  R = 0.0
  for j in range(0, N):
    R += A[j] * B[j]
  return R


avg_time = 0
for i in range(0, times):
  
  if i >= times / 2 - 1:
     start = time.monotonic()
     R = dp(N, A, B)
     end = time.monotonic()
     avg_time += (end -start)
  else:
     R = dp(N, A, B)
  
avg_time = avg_time / (times / 2)
bandwidth = 2 * 4 * N / avg_time / 1000000000
flop = (2 * (times / 2) * N) / avg_time / 100000000
print("N: {0}, time: {1} s, bandwidth: {2} GB/s, flops: {3}".format(N, avg_time, bandwidth, flop))
#print("N is ",N)
#print("times: ",times)
