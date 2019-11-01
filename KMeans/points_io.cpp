#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "points_io.h"

#define BUF_SIZE 256

int load_points(char* filename, float** points, int* n)
{
	FILE* file;
	char buf[BUF_SIZE];
	int index, count;
	float* point;

	if (fopen_s(&file, filename, "r") != 0)
	{
		fprintf(stderr, "Error while opening a file: %s\n", filename);
		exit(1);
	}

	if (fgets(buf, sizeof(buf), file) == NULL || sscanf_s(buf, "%d", &count) == 0)
	{
		fclose(file);
		return 1;
	}

	*points = (float*) calloc(count * DIM, sizeof(float));
	index = 0;

	while (index < count && fgets(buf, sizeof(buf), file) != NULL)
	{
		point = *points + DIM * index++;
		if (!sscanf_s(buf, "%f %f %f", point, point + 1, point + 2))
		{
			fclose(file);
			free(points);
			return 1;
		}
	}

	if (ferror(file))
	{
		perror("Error while reading a file");
		
		fclose(file);
		free(points);
		exit(EXIT_FAILURE);
	}

	*n = count;
	fclose(file);
	return 0;
}


