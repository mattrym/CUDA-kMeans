#ifndef GPU_KMEANS_CUH_
#define GPU_KMEANS_CUH_

#define THREADS_PER_BLOCK 1024

void gpu_kmeans(int n, int k, float max_delta, float* points, float* means, int* assignments);

#endif