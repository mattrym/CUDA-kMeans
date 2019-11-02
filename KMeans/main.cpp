#include <stdio.h>
#include <stdlib.h>

#include "points.h"
#include "points_io.h"
#include "gpu_kmeans.h"

int main(int argc, char* argv[])
{
	char* filename;
	
	int i, n;
	float* points;
	float* means;
	int* asgns;

	const int k = 3;

	if (argc != 2)
	{
		fprintf(stderr, "Usage: %s FILENAME\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	filename = argv[1];

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

	gpu_kmeans(n, k, 0.0, points, means, asgns);

	for (i = 0; i < n; ++i)
	{
		printf("%.4f %.4f %.4f %d\n", (points + i * DIM)[0], (points + i * DIM)[1], (points + i * DIM)[2], asgns[i]);
	}

	free(points);
	free(means);
	free(asgns);

	exit(EXIT_SUCCESS);
}


