#ifndef GPU_KMEANS_CUH_
#define GPU_KMEANS_CUH_

#define THREADS_PER_BLOCK 1024

void gpu_kmeans(const int n, const int k, const float max_delta, const float* points, float* means, int* assignments);

#endif