#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>
#include <float.h>

int index3d(int x, int y, int z, int ny, int nz) {
    return (x * ny + y) * nz + z;
}

void d_point_stencil(double *current, double *next, int nx, int ny, int nz, int d) {
    int radius = (d - 1) / 6;

    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            for (int z = 0; z < nz; z++) {
                int index = index3d(x, y, z, ny, nz);
                double sum = current[index];

                if (x < radius || x >= nx - radius ||
                    y < radius || y >= ny - radius ||
                    z < radius || z >= nz - radius) {
                    next[index] = current[index];
                    continue;
                }

                for (int distance = 1; distance <= radius; distance++) {
                    sum += current[index3d(x - distance, y, z, ny, nz)];
                    sum += current[index3d(x + distance, y, z, ny, nz)];
                    sum += current[index3d(x, y - distance, z, ny, nz)];
                    sum += current[index3d(x, y + distance, z, ny, nz)];
                    sum += current[index3d(x, y, z - distance, ny, nz)];
                    sum += current[index3d(x, y, z + distance, ny, nz)];
                }

                next[index] = sum / d;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank;
    int comm_size;
    int d, ppn, px, py, pz, nx, ny, nz, T, F, seed;
    int arrSize;
    double isovalue;
    double **data;
    double **next;
    double *data_storage;
    double *next_storage;
    double start_time;
    double end_time;
    double local_time;
    double max_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (argc != 13) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s d ppn px py pz nx ny nz T seed F isovalue\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    d = atoi(argv[1]);
    ppn = atoi(argv[2]);
    px = atoi(argv[3]);
    py = atoi(argv[4]);
    pz = atoi(argv[5]);
    nx = atoi(argv[6]);
    ny = atoi(argv[7]);
    nz = atoi(argv[8]);
    T = atoi(argv[9]);
    seed = atoi(argv[10]);
    F = atoi(argv[11]);
    isovalue = atof(argv[12]);
    arrSize = nx * ny * nz;

    if (px * py * pz != comm_size) {
        if (rank == 0) {
            fprintf(stderr, "px * py * pz must equal the number of MPI processes.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (d >= nx || d >= ny || d >= nz) {
        if (rank == 0) {
            fprintf(stderr, "d must be less than nx, ny, and nz.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (d < 7 || (d - 1) % 6 != 0) {
        if (rank == 0) {
            fprintf(stderr, "d must be 7, 13, 19, ...\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    
    data = malloc((size_t)F * sizeof *data);
    next = malloc((size_t)F * sizeof *next);
    data_storage = malloc((size_t)F * arrSize * sizeof *data_storage);
    next_storage = malloc((size_t)F * arrSize * sizeof *next_storage);
    if (data == NULL || next == NULL || data_storage == NULL || next_storage == NULL) {
        fprintf(stderr, "Rank %d could not allocate data array.\n", rank);
        free(data);
        free(next);
        free(data_storage);
        free(next_storage);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    for (int i = 0; i < F; i++) {
        data[i] = data_storage + (size_t)i * arrSize;
        next[i] = next_storage + (size_t)i * arrSize;
    }

    srand(seed);
    for (int i = 0; i < F; i++) {
        for (int j = 0; j < arrSize; j++) {
            data[i][j] = (double)rand() * (rank + 1) / (110426.0 + i + j);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    for (int t = 0; t < T; t++) {
        for (int i = 0; i < F; i++) {
            d_point_stencil(data[i], next[i], nx, ny, nz, d);
        }

        double **temp = data;
        data = next;
        next = temp;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    local_time = end_time - start_time;

    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("max_time=%f\n", max_time);
    }

    free(next_storage);
    free(data_storage);
    free(next);
    free(data);

    MPI_Finalize();
    
    return 0;
}
