#include <stdio.h>
#include <stdlib.h>

#include "points.h"
#include "points_io.h"

#include "cpu_kmeans.h"
#include "gpu_kmeans.h"

int main(int argc, char* argv[])
{
	char* filename;
	
	int i, n, k;
	float max_delta;

	float* points;
	float* means;
	int* asgns;

	if (argc != 4)
	{
		fprintf(stderr, "Usage: %s FILENAME K MAX_DELTA\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	
	filename = argv[1];
	k = atoi(argv[2]);
	max_delta = atof(argv[3]);

	if (k < 2)
	{
		fprintf(stderr, "k value should be higher than one");
		exit(EXIT_FAILURE);
	}
	
	if (load_points(filename, &points, &n))
	{
		fprintf(stderr, "Error while processing file (invalid format): %s\n", filename);
		exit(EXIT_FAILURE);
	}

	means = (float*)calloc(k * DIM, sizeof(float));
	if (!means)
	{
		perror("Error while allocating memory for cluster centroids");
		exit(1);
	}

	asgns = (int*)calloc(n, sizeof(int));
	if (!asgns)
	{
		perror("Error while allocating memory for assignments");
		exit(1);
	}

	cpu_kmeans(n, k, max_delta, points, means, asgns);
	for (i = 0; i < k; ++i)
	{
		printf("c[%d]: %.4f %.4f %.4f\n", i, means[i * DIM], means[i * DIM + 1], means[i * DIM + 2]);
	}

	gpu_kmeans(n, k, max_delta, points, means, asgns);
	for (i = 0; i < k; ++i)
	{
		printf("c[%d]: %.4f %.4f %.4f\n", i, means[i * DIM], means[i * DIM + 1], means[i * DIM + 2]);
	}

	free(points);
	free(means);
	free(asgns);

	exit(EXIT_SUCCESS);
}


