#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "points.h"
#include "gpu_kmeans.h"

inline void check_cuda_error(cudaError_t cuda_status, const char* file, int line)
{
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "CUDA error (%s:%d): %s", file, line, cudaGetErrorString(cuda_status));
		exit(EXIT_FAILURE);
	}
}
#define CUDA_SAFE(cuda_status) check_cuda_error(cuda_status, __FILE__, __LINE__)

__device__ void init_shared_centroids(int k, float* s_means, float* g_means)
{
	int mean_idx;

	if (threadIdx.x < k) {
		mean_idx = threadIdx.x * DIM;

		s_means[mean_idx] = g_means[mean_idx];
		s_means[mean_idx + 1] = g_means[mean_idx + 1];
		s_means[mean_idx + 2] = g_means[mean_idx + 2];
	}
	__syncthreads();
}

__device__ float distance(float* p1, float* p2)
{
	return (p2[0] - p1[0]) * (p2[0] - p1[0])
		+ (p2[1] - p1[1]) * (p2[1] - p1[1])
		+ (p2[2] - p1[2]) * (p2[2] - p1[2]);
}

__device__ void assign_cluster(int k, float* point, float* means, int* asgns, int* n_asgns)
{
	float dist, min_dist;
	int cluster, min_cluster;

	int point_idx = threadIdx.x;

	n_asgns[point_idx] = 0;
	min_dist = FLT_MAX;
	min_cluster = -1;

	for (cluster = 0; cluster < k; ++cluster)
	{
		dist = distance(point, means + cluster * DIM);
		if (dist < min_dist)
		{
			min_dist = dist;
			min_cluster = cluster;
		}
	}

	if (min_cluster != asgns[point_idx])
	{
		n_asgns[point_idx] = 1;
		asgns[point_idx] = min_cluster;
	}
}

__device__ void reduce_clusters(int k, float* point, int asgn, float* s_sums, int* s_counts, int* s_new_asgns, float* sums, int* counts, int* new_asgns)
{
	int cluster, cluster_idx;
	int offset, offset_idx;

	const int thread_idx = threadIdx.x;

	for (offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if (threadIdx.x < offset)
		{
			s_new_asgns[threadIdx.x] += s_new_asgns[offset + threadIdx.x];
		}
		__syncthreads();
	}

	if (!threadIdx.x)
	{
		new_asgns[blockIdx.x] = s_new_asgns[0];
	}

	for (cluster = 0; cluster < k; ++cluster)
	{
		s_sums[threadIdx.x * DIM] = cluster == asgn ? point[0] : 0;
		s_sums[threadIdx.x * DIM + 1] = cluster == asgn ? point[1] : 0;
		s_sums[threadIdx.x * DIM + 2] = cluster == asgn ? point[2] : 0;

		s_counts[threadIdx.x] = cluster == asgn ? 1 : 0;

		__syncthreads();

		for (offset = blockDim.x / 2; offset > 0; offset >>= 1)
		{
			if (threadIdx.x < offset)
			{
				offset_idx = offset + threadIdx.x;

				s_sums[threadIdx.x * DIM] += s_sums[offset_idx * DIM];
				s_sums[threadIdx.x * DIM + 1] += s_sums[offset_idx * DIM + 1];
				s_sums[threadIdx.x * DIM + 2] += s_sums[offset_idx * DIM + 2];

				s_counts[threadIdx.x] += s_counts[offset + threadIdx.x];
			}
			__syncthreads();
		}

		if (!threadIdx.x)
		{
			cluster_idx = blockIdx.x * k + cluster;

			sums[cluster_idx * DIM] = s_sums[0];
			sums[cluster_idx * DIM + 1] = s_sums[1];
			sums[cluster_idx * DIM + 2] = s_sums[2];

			counts[cluster_idx] = s_counts[0];
		}
		__syncthreads();
	}
}

__global__ void assign_clusters(int n, int k, float* points, float* means, float* sums, int* counts, int* asgns, int* new_asgns)
{
	extern __shared__ int shared_memory[];

	int point_idx, asgn;
	float point[DIM];

	int cluster, cluster_idx;
	int offset, offset_idx;

	float* s_means = (float*)shared_memory;				// sizeof(float) * k * DIM
	float* s_sums = (float*)(s_means + k * DIM);			// sizeof(float) * blockDim.x * DIM
	int* s_asgns = (int*)(s_sums + blockDim.x);			// sizeof(int) * blockDim.x
	int* s_counts = (int*)(s_asgns + blockDim.x);			// sizeof(int) * blockDim.x
	int* s_new_asgns = (int*)(s_counts + blockDim.x);		// sizeof(int) * n

	point_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (point_idx >= n)
	{
		return;
	}

	point[0] = (points + point_idx * DIM)[0];
	point[1] = (points + point_idx * DIM)[1];
	point[2] = (points + point_idx * DIM)[2];

	init_shared_centroids(k, s_means, means);
	assign_cluster(k, point, s_means, s_asgns, s_new_asgns);
	__syncthreads();

	asgn = asgns[point_idx] = s_asgns[threadIdx.x];
	reduce_clusters(k, point, asgn, s_sums, s_counts, s_new_asgns, sums, counts, new_asgns);
}

__global__ void calculate_means(int n, int k, float* means, float* delta, float* sums, int* counts, int* new_asgns)
{
	extern __shared__ int shared_memory[];

	int idx, offset, offset_idx, count;
	const int blocks = blockDim.x / k;

	float* s_sums = (float*)shared_memory;
	int* s_counts = (int*)(s_sums + blockDim.x * DIM);
	int* s_new_asgns = (int*)(s_counts + blockDim.x);

	idx = threadIdx.x * DIM;

	s_sums[idx] = sums[idx];
	s_sums[idx + 1] = sums[idx + 1];
	s_sums[idx + 2] = sums[idx + 2];

	s_counts[threadIdx.x] = counts[threadIdx.x];
	if (threadIdx.x < blocks)
	{
		s_new_asgns[threadIdx.x] = new_asgns[threadIdx.x];
	}
	__syncthreads();


	for (offset = blockDim.x / 2; offset >= k; offset >>= 1)
	{
		if (threadIdx.x < offset)
		{
			offset_idx = (offset + threadIdx.x) * DIM;

			s_sums[idx] += s_sums[offset_idx];
			s_sums[idx + 1] += s_sums[offset_idx + 1];
			s_sums[idx + 2] += s_sums[offset_idx + 2];

			s_counts[threadIdx.x] += s_counts[offset + threadIdx.x];
		}
		__syncthreads();
	}

	if (threadIdx.x < blocks)
	{
		for (offset = blocks / 2; offset > 0; offset >>= 1)
		{
			if (threadIdx.x < offset)
			{
				s_new_asgns[threadIdx.x] += s_new_asgns[offset + threadIdx.x];
			}
			__syncthreads();
		}
	}

	if (threadIdx.x < k)
	{
		count = s_counts[threadIdx.x] > 0 ? s_counts[threadIdx.x] : 1;

		means[idx] = s_sums[idx] / count;
		means[idx + 1] = s_sums[idx + 1] / count;
		means[idx + 2] = s_sums[idx + 2] / count;
	}

	if (threadIdx.x == 0)
	{
		*delta = s_new_asgns[0] * 1.0 / n;
	}
}

void kmeans_gpu(int n, int k, float max_delta, float* input_points, float* output_means, int* output_asgns)
{
	float *points, *means, *sums;
	int *asgns, *new_asgns, *counts;
	float *d_delta, h_delta;

	const int threads = 1024;
	const int blocks = (n + threads - 1) / threads;

	const int ac_mem_size =
		k * DIM * sizeof(float) +
		threads * DIM * sizeof(float) +
		threads * sizeof(int) +
		threads * sizeof(int) +
		blocks * sizeof(int);
	const int cm_mem_size =
		blocks * k * DIM * sizeof(float) +
		blocks * k * sizeof(int) +
		blocks * sizeof(int);

	CUDA_SAFE(cudaSetDevice(0));

	CUDA_SAFE(cudaMalloc((void**)&points, n * DIM * sizeof(float)));
	CUDA_SAFE(cudaMalloc((void**)&means, k * DIM * sizeof(float)));
	CUDA_SAFE(cudaMalloc((void**)&asgns, n * sizeof(int)));

	CUDA_SAFE(cudaMalloc((void**)&sums, blocks * k * DIM * sizeof(float)));
	CUDA_SAFE(cudaMalloc((void**)&counts, blocks * k * sizeof(int)));
	CUDA_SAFE(cudaMalloc((void**)&new_asgns, blocks * sizeof(int)));

	CUDA_SAFE(cudaMalloc((void**)&d_delta, sizeof(float)));

	CUDA_SAFE(cudaMemcpy(points, input_points, n * DIM * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(means, input_points, k * DIM * sizeof(float), cudaMemcpyHostToDevice));

	h_delta = 1;
	while (h_delta > max_delta)
	{
		assign_clusters <<<blocks, threads, ac_mem_size>> >(n, k, points, means, sums, counts, asgns, new_asgns);
		CUDA_SAFE(cudaDeviceSynchronize());

		calculate_means <<<1, blocks * k, cm_mem_size>>>(n, k, means, d_delta, sums, counts, new_asgns);
		CUDA_SAFE(cudaDeviceSynchronize());

		CUDA_SAFE(cudaMemcpy(&h_delta, d_delta, sizeof(float), cudaMemcpyDeviceToHost));
	}

	CUDA_SAFE(cudaMemcpy(output_means, means, k * DIM * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_SAFE(cudaMemcpy(output_asgns, asgns, n * sizeof(int), cudaMemcpyDeviceToHost));

	cudaFree(points);
	cudaFree(means);
	cudaFree(asgns);

	cudaFree(sums);
	cudaFree(counts);
	cudaFree(new_asgns);

	CUDA_SAFE(cudaDeviceReset());
}