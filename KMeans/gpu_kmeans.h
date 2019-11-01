#ifndef GPU_KMEANS_CUH_
#define GPU_KMEANS_CUH_

void kmeans_gpu(int n, int k, float max_delta, float* input_points, float* output_means, int* output_asgns);

#endif