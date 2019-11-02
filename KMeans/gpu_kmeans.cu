#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "points.h"
#include "gpu_kmeans.h"

inline void check_cuda_error(const cudaError_t cuda_status, const char* file, const int line)
{
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "CUDA error (%s:%d): %s", file, line, cudaGetErrorString(cuda_status));
		exit(EXIT_FAILURE);
	}
}
#define CUDA_SAFE(cuda_status) check_cuda_error(cuda_status, __FILE__, __LINE__)

__device__ void init_shared_means(const int k, const float* means, float* s_means)
{
	int mean_index;

	if (threadIdx.x < k) {
		mean_index = threadIdx.x * DIM;

		s_means[mean_index] = means[mean_index];
		s_means[mean_index + 1] = means[mean_index + 1];
		s_means[mean_index + 2] = means[mean_index + 2];
	}
}

__device__ float gpu_square_distance(const float* p1, const float* p2)
{
	return (p2[0] - p1[0]) * (p2[0] - p1[0])
		+ (p2[1] - p1[1]) * (p2[1] - p1[1])
		+ (p2[2] - p1[2]) * (p2[2] - p1[2]);
}

__device__ void assign_cluster(const int k, const float* point, const float* means, int* assignments, int* changes)
{
	float distance, min_distance;
	int cluster, best_cluster;

	min_distance = FLT_MAX;
	best_cluster = -1;

	for (cluster = 0; cluster < k; ++cluster)
	{
		distance = gpu_square_distance(point, means + cluster * DIM);
		if (distance < min_distance)
		{
			min_distance = distance;
			best_cluster = cluster;
		}
	}

	changes[threadIdx.x] = 0;
	if (best_cluster != assignments[threadIdx.x])
	{
		assignments[threadIdx.x] = best_cluster;
		changes[threadIdx.x] = 1;
	}
}

__device__ void reduce_changes(const int window, int* changes)
{
	int offset, last_offset = window;

	for (offset = window >> 1; offset > 0; offset >>= 1)
	{
		if (threadIdx.x < offset)
		{
			changes[threadIdx.x] += changes[offset + threadIdx.x];
		}
		if (threadIdx.x == offset && last_offset & 1)
		{
			changes[threadIdx.x] = changes[last_offset - 1];
			offset += 1;
		}

		last_offset = offset;
		__syncthreads();
	}
}

__device__ void reduce_cluster_points(const int window, const float* point, const int include, float* sums, int* counts)
{
	int offset, last_offset;
	int offset_index, offset_sum_index;

	const int index = threadIdx.x;
	const int sum_index = index * DIM;

	sums[sum_index] = include ? point[0] : 0;
	sums[sum_index + 1] = include ? point[1] : 0;
	sums[sum_index + 2] = include ? point[2] : 0;

	counts[index] = include ? 1 : 0;

	__syncthreads();

	last_offset = window;
	for (offset = window >> 1; offset > 0; last_offset = offset, offset >>= 1)
	{
		offset_index = offset + index;
		offset_sum_index = offset_index * DIM;

		if (threadIdx.x < offset)
		{
			sums[sum_index] += sums[offset_sum_index];
			sums[sum_index + 1] += sums[offset_sum_index + 1];
			sums[sum_index + 2] += sums[offset_sum_index + 2];

			counts[index] += counts[offset_index];
		}
		if (threadIdx.x == offset && last_offset & 1)
		{
			sums[sum_index] = sums[offset_sum_index];
			sums[sum_index + 1] = sums[offset_sum_index + 1];
			sums[sum_index + 2] = sums[offset_sum_index + 2];

			counts[index] = counts[offset_index];

			offset += 1;
		}

		__syncthreads();
	}
}

__device__ void export_cluster_sums(const int k, const int cluster, const float* s_sums, const int* s_counts, float* sums, int* counts)
{
	const int cluster_index = blockIdx.x * k + cluster;

	if (threadIdx.x == 0)
	{
		sums[cluster_index * DIM] = s_sums[0];
		sums[cluster_index * DIM + 1] = s_sums[1];
		sums[cluster_index * DIM + 2] = s_sums[2];

		counts[cluster_index] = s_counts[0];
	}
}

__global__ void assign_clusters(const int n, const int k, const float* points, const float* means, float* sums, int* counts, int* assignments, int* changes)
{
	extern __shared__ int shared_memory[];

	int assignment, cluster;
	float point[DIM];

	float* s_means = (float*)shared_memory;					// sizeof(float) * k * DIM (for each cluster)
	float* s_sums = (float*)(s_means + k * DIM);			// sizeof(float) * blockDim.x * DIM (for each point)
	int* s_counts = (int*)(s_sums + blockDim.x * DIM);		// sizeof(int) * blockDim.x (for each point)
	int* s_assignments = (int*)(s_counts + blockDim.x);		// sizeof(int) * blockDim.x (for each point)
	int* s_changes = (int*)(s_assignments + blockDim.x);	// sizeof(int) * blockDim.x (for each point)

	const int window = (blockIdx.x + 1) * blockDim.x > n ? n % blockDim.x : blockDim.x;
	const int point_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (point_index >= n)
	{
		return;
	}

	point[0] = (points + point_index * DIM)[0];
	point[1] = (points + point_index * DIM)[1];
	point[2] = (points + point_index * DIM)[2];

	init_shared_means(k, means, s_means);
	__syncthreads();

	assign_cluster(k, point, s_means, s_assignments, s_changes);
	assignments[point_index] = assignment = s_assignments[threadIdx.x];
	__syncthreads();

	for (cluster = 0; cluster < k; ++cluster)
	{
		reduce_cluster_points(window, point, assignment == cluster, s_sums, s_counts);
		export_cluster_sums(k, cluster, s_sums, s_counts, sums, counts);
		__syncthreads();
	}

	reduce_changes(blockDim.x, s_changes);
	if (threadIdx.x == 0)
	{
		changes[blockIdx.x] = s_changes[0];
	}	
}

__device__ void reduce_means(const int blocks, const int k, const float* sum, const int count, float* sums, int* counts)
{
	int offset, last_offset;
	int offset_index, offset_sum_index;

	const int index = threadIdx.x;
	const int sum_index = index * DIM;

	sums[sum_index] = sum[0];
	sums[sum_index + 1] = sum[1];
	sums[sum_index + 2] = sum[2];
	counts[index] = count;

	__syncthreads();

	last_offset = blocks;
	for (offset = blocks >> 1; offset > 0; last_offset = offset, offset >>= 1)
	{
		offset_index = offset * k + threadIdx.x;
		offset_sum_index = offset_index * DIM;

		if (threadIdx.x < offset * k)
		{
			sums[sum_index] += sums[offset_sum_index];
			sums[sum_index + 1] += sums[offset_sum_index + 1];
			sums[sum_index + 2] += sums[offset_sum_index + 2];

			counts[index] += counts[offset_index];
		}
		if (threadIdx.x == offset * k && last_offset & 1)
		{
			sums[sum_index] = sums[offset_sum_index];
			sums[sum_index + 1] = sums[offset_sum_index + 1];
			sums[sum_index + 2] = sums[offset_sum_index + 2];

			counts[index] = counts[offset_index];

			offset += 1;
		}

		__syncthreads();
	}
}

__device__ void export_means(const int k, const float* s_sums, const int* s_counts, float* means)
{
	int count;
	const int index = threadIdx.x;
	const int sum_index = index * DIM;

	if (index < k)
	{
		count = s_counts[index] > 0 ? s_counts[index] : 1;

		means[sum_index] = s_sums[sum_index] / count;
		means[sum_index + 1] = s_sums[sum_index + 1] / count;
		means[sum_index + 2] = s_sums[sum_index + 2] / count;
	}
}

__device__ void export_delta(const int n, const int blocks, const int* changes, int* s_changes, float* delta)
{
	const int index = threadIdx.x;

	if (index < blocks)
	{
		s_changes[index] = changes[index];
		__syncthreads();

		reduce_changes(blocks, s_changes);
		if (threadIdx.x == 0)
		{
			*delta = s_changes[0] * 1.0 / n;
		}
	}
}

__global__ void calculate_means(const int n, const int k, const float* sums, const int* counts, const int* changes, float* means, float* delta)
{
	extern __shared__ int shared_memory[];

	float* s_sums = (float*)shared_memory;
	int* s_counts = (int*)(s_sums + blockDim.x * DIM);
	int* s_changes = (int*)(s_counts + blockDim.x);

	const float* sum = sums + threadIdx.x * DIM;
	const int count = counts[threadIdx.x];
	const int blocks = blockDim.x / k;

	reduce_means(blocks, k, sum, count, s_sums, s_counts);
	export_means(k, s_sums, s_counts, means);
	export_delta(n, blocks, changes, s_changes, delta);
}

void gpu_kmeans(const int n, const int k, const float max_delta, const float* input_points, float* output_means, int* output_assignments)
{
	int *counts, *assignments, *changes;
	float *points, *means, *sums;
	float *device_delta, host_delta;

	const int threads = THREADS_PER_BLOCK;
	const int blocks = (n + threads - 1) / threads;

	const int ac_mem_size = (k * DIM) * sizeof(float) + (threads * DIM) * sizeof(float) + (3 * threads) * sizeof(int);
	const int cm_mem_size = (blocks * k * DIM) * sizeof(float) + (blocks * k) * sizeof(int) + blocks * sizeof(int);

	CUDA_SAFE(cudaSetDevice(0));

	CUDA_SAFE(cudaMalloc((void**)&points, n * DIM * sizeof(float)));
	CUDA_SAFE(cudaMalloc((void**)&means, k * DIM * sizeof(float)));
	CUDA_SAFE(cudaMalloc((void**)&assignments, n * sizeof(int)));

	CUDA_SAFE(cudaMalloc((void**)&sums, blocks * k * DIM * sizeof(float)));
	CUDA_SAFE(cudaMalloc((void**)&counts, blocks * k * sizeof(int)));
	CUDA_SAFE(cudaMalloc((void**)&changes, blocks * sizeof(int)));

	CUDA_SAFE(cudaMalloc((void**)&device_delta, sizeof(float)));

	CUDA_SAFE(cudaMemcpy(points, input_points, n * DIM * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(means, input_points, k * DIM * sizeof(float), cudaMemcpyHostToDevice));

	host_delta = 1;
	while (host_delta > max_delta)
	{
		assign_clusters<<<blocks, threads, ac_mem_size>>>(n, k, points, means, sums, counts, assignments, changes);
		CUDA_SAFE(cudaDeviceSynchronize());

		calculate_means<<<1, blocks * k, cm_mem_size>>>(n, k, sums, counts, changes, means, device_delta);
		CUDA_SAFE(cudaDeviceSynchronize());

		CUDA_SAFE(cudaMemcpy(&host_delta, device_delta, sizeof(float), cudaMemcpyDeviceToHost));
	}

	CUDA_SAFE(cudaMemcpy(output_means, means, k * DIM * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_SAFE(cudaMemcpy(output_assignments, assignments, n * sizeof(int), cudaMemcpyDeviceToHost));

	cudaFree(points);
	cudaFree(means);
	cudaFree(assignments);

	cudaFree(sums);
	cudaFree(counts);
	cudaFree(changes);

	CUDA_SAFE(cudaDeviceReset());
}