#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <windows.h>

#include "points.h"
#include "cpu_kmeans.h"

float cpu_square_distance(const float* p1, const float* p2)
{
	return (p2[0] - p1[0]) * (p2[0] - p1[0])
		+ (p2[1] - p1[1]) * (p2[1] - p1[1])
		+ (p2[2] - p1[2]) * (p2[2] - p1[2]);
}

int find_best_cluster(const int k, const float* point, const float* means)
{
	int cluster, best_cluster;
	float distance, min_distance;

	min_distance = FLT_MAX;
	best_cluster = -1;

	for (cluster = 0; cluster < k; ++cluster)
	{
		distance = cpu_square_distance(point, means + cluster * DIM);
		if (distance < min_distance)
		{
			min_distance = distance;
			best_cluster = cluster;
		}
	}

	return best_cluster;
}

void initialize_means(const int k, const float* points, float* means)
{
	int cluster, mean_index;

	for (cluster = 0; cluster < k; ++cluster)
	{
		mean_index = cluster * DIM;

		means[mean_index] = points[mean_index];
		means[mean_index + 1] = points[mean_index + 1];
		means[mean_index + 2] = points[mean_index + 2];
	}
}

void accumulate_point(const int cluster, const float* point, float* cluster_sums, int* cluster_size)
{
	cluster_sums[cluster * DIM] += point[0];
	cluster_sums[cluster * DIM + 1] += point[1];
	cluster_sums[cluster * DIM + 2] += point[2];

	cluster_size[cluster]++;
}

void recalculate_means(const int k, float* cluster_sums, int* cluster_size, float* means)
{
	int cluster; 

	for (cluster = 0; cluster < k; ++cluster)
	{
		means[cluster * DIM] = cluster_sums[cluster * DIM] / cluster_size[cluster];
		means[cluster * DIM + 1] = cluster_sums[cluster * DIM + 1] / cluster_size[cluster];
		means[cluster * DIM + 2] = cluster_sums[cluster * DIM + 2] / cluster_size[cluster];
	}
}

void cpu_kmeans(int n, int k, float max_delta, float* points, float* means, int* assignments)
{
	int point_index, best_cluster, delta = n;
	int i, it = 0;

	LARGE_INTEGER freq, start, end;
	double elapsed_time;

	float* cluster_sums = (float*)calloc(k * DIM, sizeof(float));
	int* cluster_size = (int*)calloc(k, sizeof(int));

	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);

	initialize_means(k, points, means);
	
	while (delta > max_delta * n)
	{
		it++;
		delta = 0;
		for (point_index = 0; point_index < n; ++point_index)
		{
			best_cluster = find_best_cluster(k, points + point_index * DIM, means);

			if (best_cluster != assignments[point_index])
			{
				assignments[point_index] = best_cluster;
				delta++;
			}

			accumulate_point(best_cluster, points + point_index * DIM, cluster_sums, cluster_size);
		}

		recalculate_means(k, cluster_sums, cluster_size, means);
		
		memset(cluster_sums, 0, k * DIM * sizeof(float));
		memset(cluster_size, 0, k * sizeof(int));
	}

	QueryPerformanceCounter(&end);

	elapsed_time = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
	printf("\nCPU time: %.7g ms \t CPU iterations: %d\n", elapsed_time, it);

	free(cluster_sums);
	free(cluster_size);
}