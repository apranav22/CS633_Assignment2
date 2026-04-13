/*
 * ass2.c — MPI parallel stencil + isovalue counter
 *
 * Distributes F scalar fields over a (px × py × pz) process grid.  Each
 * process owns a local (nx × ny × nz) tile.  At every time step it:
 *   1. Exchanges one-layer halos with its six face neighbours.
 *   2. Counts directed edges that cross the isovalue surface.
 *   3. Applies a d-point 6-axis averaging stencil (radius = (d-1)/6).
 *
 * Indexing convention throughout: x varies fastest, z slowest (x-major).
 * The j-th rand() call is assigned to the cell at logical position j decoded
 * in x-fastest order, so initialization is consistent across implementations.
 *
 * Command-line usage:
 *   ./ass2 d ppn px py pz nx ny nz T seed F isovalue
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include <math.h>
#include <float.h>

// Neighbor ranks for this process in the 3D process grid.
// A field value of MPI_PROC_NULL means the process sits on a global boundary.
typedef struct {
    int west, east, south, north, back, front;
} ProcessGrid;

// Halo storage for boundary slabs exchanged with neighboring processes.
// Each axis has a send buffer (data this process sends out) and a receive
// buffer (data arriving from the neighbour).  x_count/y_count/z_count give
// the number of doubles in the corresponding slab pair.
typedef struct {
    int radius;
    int nx, ny, nz;
    int x_count, y_count, z_count;
    int west_neighbor_exists, east_neighbor_exists;
    int south_neighbor_exists, north_neighbor_exists;
    int back_neighbor_exists, front_neighbor_exists;
    double *send_west, *send_east, *west, *east;
    double *send_south, *send_north, *south, *north;
    double *send_back, *send_front, *back, *front;
} HaloBuffers;

// Flatten a 3-D (x, y, z) index into a 1-D array offset.
// Storage order is x-major: x is the slowest-varying dimension, z the fastest.
int index3d(int x, int y, int z, int ny, int nz) {
    return (x * ny + y) * nz + z;
}

// Convert a (x, y, z) position in the process grid to a flat MPI rank.
// x varies fastest, then y, then z.
int rank_from_coords(int x, int y, int z, int px, int py) {
    return (z * py + y) * px + x;
}

// Populate *grid with the MPI ranks of the six face neighbours for this process.
// Processes on the global domain boundary get MPI_PROC_NULL for the missing face.
void setup_process_grid(ProcessGrid *grid, int rank, int px, int py, int pz) {
    // Decompose the flat rank into (x, y, z) coordinates, x varying fastest.
    int x = rank % px;
    int y = (rank / px) % py;
    int z = rank / (px * py);

    grid->west = (x > 0) ? rank_from_coords(x - 1, y, z, px, py) : MPI_PROC_NULL;
    grid->east = (x < px - 1) ? rank_from_coords(x + 1, y, z, px, py) : MPI_PROC_NULL;
    grid->south = (y > 0) ? rank_from_coords(x, y - 1, z, px, py) : MPI_PROC_NULL;
    grid->north = (y < py - 1) ? rank_from_coords(x, y + 1, z, px, py) : MPI_PROC_NULL;
    grid->back = (z > 0) ? rank_from_coords(x, y, z - 1, px, py) : MPI_PROC_NULL;
    grid->front = (z < pz - 1) ? rank_from_coords(x, y, z + 1, px, py) : MPI_PROC_NULL;
}

// Allocate all twelve halo buffers (six send + six receive) for a local tile of
// size (nx × ny × nz) with the given stencil radius.
// Returns 1 on success, 0 if any allocation fails.
int allocate_halos(HaloBuffers *halo, int radius, int nx, int ny, int nz) {
    halo->radius = radius;
    halo->nx = nx;
    halo->ny = ny;
    halo->nz = nz;
    // Each slab is `radius` layers deep, spanning the full face of the tile.
    halo->x_count = radius * ny * nz;
    halo->y_count = radius * nx * nz;
    halo->z_count = radius * nx * ny;
    halo->west_neighbor_exists = 0;
    halo->east_neighbor_exists = 0;
    halo->south_neighbor_exists = 0;
    halo->north_neighbor_exists = 0;
    halo->back_neighbor_exists = 0;
    halo->front_neighbor_exists = 0;

    halo->send_west = malloc((size_t)halo->x_count * sizeof *halo->send_west);
    halo->send_east = malloc((size_t)halo->x_count * sizeof *halo->send_east);
    halo->west = malloc((size_t)halo->x_count * sizeof *halo->west);
    halo->east = malloc((size_t)halo->x_count * sizeof *halo->east);
    halo->send_south = malloc((size_t)halo->y_count * sizeof *halo->send_south);
    halo->send_north = malloc((size_t)halo->y_count * sizeof *halo->send_north);
    halo->south = malloc((size_t)halo->y_count * sizeof *halo->south);
    halo->north = malloc((size_t)halo->y_count * sizeof *halo->north);
    halo->send_back = malloc((size_t)halo->z_count * sizeof *halo->send_back);
    halo->send_front = malloc((size_t)halo->z_count * sizeof *halo->send_front);
    halo->back = malloc((size_t)halo->z_count * sizeof *halo->back);
    halo->front = malloc((size_t)halo->z_count * sizeof *halo->front);

    return halo->send_west && halo->send_east && halo->west && halo->east &&
           halo->send_south && halo->send_north && halo->south && halo->north &&
           halo->send_back && halo->send_front && halo->back && halo->front;
}

// Release all halo buffers allocated by allocate_halos().
void free_halos(HaloBuffers *halo) {
    free(halo->send_west);
    free(halo->send_east);
    free(halo->west);
    free(halo->east);
    free(halo->send_south);
    free(halo->send_north);
    free(halo->south);
    free(halo->north);
    free(halo->send_back);
    free(halo->send_front);
    free(halo->back);
    free(halo->front);
}

// Copy the neighbour-existence flags from the process grid into the halo struct
// so the stencil and isovalue routines can test them without a pointer to grid.
// MPI_PROC_NULL marks a global boundary with no neighboring process.
void set_halo_neighbors(HaloBuffers *halo, ProcessGrid *grid) {
    halo->west_neighbor_exists = grid->west != MPI_PROC_NULL;
    halo->east_neighbor_exists = grid->east != MPI_PROC_NULL;
    halo->south_neighbor_exists = grid->south != MPI_PROC_NULL;
    halo->north_neighbor_exists = grid->north != MPI_PROC_NULL;
    halo->back_neighbor_exists = grid->back != MPI_PROC_NULL;
    halo->front_neighbor_exists = grid->front != MPI_PROC_NULL;
}

// Pack `radius` x-slabs starting at x = start_x into buffer (row-major: y outer, z inner).
void pack_x(double *field, double *buffer, int start_x, HaloBuffers *halo) {
    int p = 0;
    for (int layer = 0; layer < halo->radius; layer++) {
        int x = start_x + layer;
        for (int y = 0; y < halo->ny; y++) {
            for (int z = 0; z < halo->nz; z++) {
                buffer[p++] = field[index3d(x, y, z, halo->ny, halo->nz)];
            }
        }
    }
}

// Pack `radius` y-slabs starting at y = start_y into buffer (row-major: x outer, z inner).
void pack_y(double *field, double *buffer, int start_y, HaloBuffers *halo) {
    int p = 0;
    for (int layer = 0; layer < halo->radius; layer++) {
        int y = start_y + layer;
        for (int x = 0; x < halo->nx; x++) {
            for (int z = 0; z < halo->nz; z++) {
                buffer[p++] = field[index3d(x, y, z, halo->ny, halo->nz)];
            }
        }
    }
}

// Pack `radius` z-slabs starting at z = start_z into buffer (row-major: x outer, y inner).
void pack_z(double *field, double *buffer, int start_z, HaloBuffers *halo) {
    int p = 0;
    for (int layer = 0; layer < halo->radius; layer++) {
        int z = start_z + layer;
        for (int x = 0; x < halo->nx; x++) {
            for (int y = 0; y < halo->ny; y++) {
                buffer[p++] = field[index3d(x, y, z, halo->ny, halo->nz)];
            }
        }
    }
}

// Exchange the boundary slabs needed for stencil reads that cross process edges.
// All six faces are handled with non-blocking Irecv/Isend pairs; receives are
// posted before sends to avoid potential deadlock on MPI implementations that
// lack sufficient internal buffering.  MPI_Waitall completes the exchange.
void exchange_halos(double *field, ProcessGrid *grid, HaloBuffers *halo) {
    MPI_Request reqs[12];
    int req_count = 0;

    // Zero receive buffers so stale data is never visible if a neighbour is absent.
    memset(halo->west, 0, (size_t)halo->x_count * sizeof *halo->west);
    memset(halo->east, 0, (size_t)halo->x_count * sizeof *halo->east);
    memset(halo->south, 0, (size_t)halo->y_count * sizeof *halo->south);
    memset(halo->north, 0, (size_t)halo->y_count * sizeof *halo->north);
    memset(halo->back, 0, (size_t)halo->z_count * sizeof *halo->back);
    memset(halo->front, 0, (size_t)halo->z_count * sizeof *halo->front);

    // Pack this process's boundary slabs into the send buffers.
    pack_x(field, halo->send_west, 0, halo);
    pack_x(field, halo->send_east, halo->nx - halo->radius, halo);
    pack_y(field, halo->send_south, 0, halo);
    pack_y(field, halo->send_north, halo->ny - halo->radius, halo);
    pack_z(field, halo->send_back, 0, halo);
    pack_z(field, halo->send_front, halo->nz - halo->radius, halo);

    // Post receives before sends so each neighbor has a matching receive ready.
    // Tag pairs are symmetric: west sends tag 0, east receives it as tag 0, etc.
    if (grid->west != MPI_PROC_NULL)
        MPI_Irecv(halo->west, halo->x_count, MPI_DOUBLE, grid->west, 1, MPI_COMM_WORLD, &reqs[req_count++]);
    if (grid->east != MPI_PROC_NULL)
        MPI_Irecv(halo->east, halo->x_count, MPI_DOUBLE, grid->east, 0, MPI_COMM_WORLD, &reqs[req_count++]);
    if (grid->south != MPI_PROC_NULL)
        MPI_Irecv(halo->south, halo->y_count, MPI_DOUBLE, grid->south, 3, MPI_COMM_WORLD, &reqs[req_count++]);
    if (grid->north != MPI_PROC_NULL)
        MPI_Irecv(halo->north, halo->y_count, MPI_DOUBLE, grid->north, 2, MPI_COMM_WORLD, &reqs[req_count++]);
    if (grid->back != MPI_PROC_NULL)
        MPI_Irecv(halo->back, halo->z_count, MPI_DOUBLE, grid->back, 5, MPI_COMM_WORLD, &reqs[req_count++]);
    if (grid->front != MPI_PROC_NULL)
        MPI_Irecv(halo->front, halo->z_count, MPI_DOUBLE, grid->front, 4, MPI_COMM_WORLD, &reqs[req_count++]);

    // Send this process's boundary slabs to the adjacent processes.
    if (grid->west != MPI_PROC_NULL)
        MPI_Isend(halo->send_west, halo->x_count, MPI_DOUBLE, grid->west, 0, MPI_COMM_WORLD, &reqs[req_count++]);
    if (grid->east != MPI_PROC_NULL)
        MPI_Isend(halo->send_east, halo->x_count, MPI_DOUBLE, grid->east, 1, MPI_COMM_WORLD, &reqs[req_count++]);
    if (grid->south != MPI_PROC_NULL)
        MPI_Isend(halo->send_south, halo->y_count, MPI_DOUBLE, grid->south, 2, MPI_COMM_WORLD, &reqs[req_count++]);
    if (grid->north != MPI_PROC_NULL)
        MPI_Isend(halo->send_north, halo->y_count, MPI_DOUBLE, grid->north, 3, MPI_COMM_WORLD, &reqs[req_count++]);
    if (grid->back != MPI_PROC_NULL)
        MPI_Isend(halo->send_back, halo->z_count, MPI_DOUBLE, grid->back, 4, MPI_COMM_WORLD, &reqs[req_count++]);
    if (grid->front != MPI_PROC_NULL)
        MPI_Isend(halo->send_front, halo->z_count, MPI_DOUBLE, grid->front, 5, MPI_COMM_WORLD, &reqs[req_count++]);

    if (req_count > 0) {
        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    }
}

// Index into a halo receive buffer at (layer, a, b), where the buffer was packed
// with dimensions (dim_a, dim_b) as the two face axes.
// Used by the stencil and isovalue routines to read ghost-cell values.
double halo_value(double *buffer, int layer, int a, int b, int dim_a, int dim_b) {
    return buffer[(layer * dim_a + a) * dim_b + b];
}

// Apply a d-point averaging stencil to every cell, writing results into `next`.
// For each cell the stencil reaches `radius` steps along each of the six axis
// directions (±x, ±y, ±z).  When a neighbour cell is outside the local tile the
// corresponding halo buffer is consulted; cells at the global boundary are simply
// omitted from the sum (neighbor_count is not incremented for missing neighbours).
void d_point_stencil(double *current, double *next, int nx, int ny, int nz, int d, HaloBuffers *halo) {
    int radius = (d - 1) / 6;

    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            for (int z = 0; z < nz; z++) {
                int index = index3d(x, y, z, ny, nz);
                double sum = current[index];
                int neighbor_count = 1;

                for (int distance = 1; distance <= radius; distance++) {
                    // -x direction: fall back to west halo if x - distance < 0.
                    if (x >= distance) {
                        sum += current[index3d(x - distance, y, z, ny, nz)];
                        neighbor_count++;
                    } else if (halo->west_neighbor_exists) {
                        // layer index within the halo: 0 = innermost layer received
                        // from the west neighbour, radius-1 = outermost.
                        sum += halo_value(halo->west, radius + x - distance, y, z, ny, nz);
                        neighbor_count++;
                    }

                    // +x direction: fall back to east halo if x + distance >= nx.
                    if (x + distance < nx) {
                        sum += current[index3d(x + distance, y, z, ny, nz)];
                        neighbor_count++;
                    } else if (halo->east_neighbor_exists) {
                        sum += halo_value(halo->east, x + distance - nx, y, z, ny, nz);
                        neighbor_count++;
                    }

                    // -y direction
                    if (y >= distance) {
                        sum += current[index3d(x, y - distance, z, ny, nz)];
                        neighbor_count++;
                    } else if (halo->south_neighbor_exists) {
                        sum += halo_value(halo->south, radius + y - distance, x, z, nx, nz);
                        neighbor_count++;
                    }

                    // +y direction
                    if (y + distance < ny) {
                        sum += current[index3d(x, y + distance, z, ny, nz)];
                        neighbor_count++;
                    } else if (halo->north_neighbor_exists) {
                        sum += halo_value(halo->north, y + distance - ny, x, z, nx, nz);
                        neighbor_count++;
                    }

                    // -z direction
                    if (z >= distance) {
                        sum += current[index3d(x, y, z - distance, ny, nz)];
                        neighbor_count++;
                    } else if (halo->back_neighbor_exists) {
                        sum += halo_value(halo->back, radius + z - distance, x, y, nx, ny);
                        neighbor_count++;
                    }

                    // +z direction
                    if (z + distance < nz) {
                        sum += current[index3d(x, y, z + distance, ny, nz)];
                        neighbor_count++;
                    } else if (halo->front_neighbor_exists) {
                        sum += halo_value(halo->front, z + distance - nz, x, y, nx, ny);
                        neighbor_count++;
                    }
                }

                next[index] = sum / neighbor_count;
            }
        }
    }
}

// Count the number of axis-aligned edges in this process's tile whose two
// endpoint values straddle the isovalue (one <= isovalue, the other > isovalue).
// Each edge is checked from the lower-index cell: the check for the +x edge of
// cell (x,y,z) is the same as the -x edge of cell (x+1,y,z), so no edge is
// double-counted.  Global boundary edges (gx=0 looking west, etc.) are skipped.
// Uses global coordinates (gx_off + x, etc.) to detect global-domain boundaries,
// and reads halo buffers for neighbour values that cross process boundaries.
long long count_isovalue_cells(double *field, int nx, int ny, int nz, double isovalue,
                                HaloBuffers *halo,
                                int gx_off, int gy_off, int gz_off,
                                int gx_max, int gy_max, int gz_max) {
    long long isovalue_count = 0;
    int radius = halo->radius;

    for (int x = 0; x < nx; x++) {
        int gx = gx_off + x;
        for (int y = 0; y < ny; y++) {
            int gy = gy_off + y;
            for (int z = 0; z < nz; z++) {
                int gz = gz_off + z;
                double center = field[index3d(x, y, z, ny, nz)];
                double v;

                // -x neighbour (skip at global west boundary)
                if (gx - 1 >= 0) {
                    v = (x - 1 >= 0) ? field[index3d(x-1, y, z, ny, nz)]
                                     : halo_value(halo->west, radius - 1, y, z, ny, nz);
                    if ((center <= isovalue && v > isovalue) || (center > isovalue && v <= isovalue))
                        isovalue_count++;
                }
                // +x neighbour (skip at global east boundary)
                if (gx + 1 < gx_max) {
                    v = (x + 1 < nx) ? field[index3d(x+1, y, z, ny, nz)]
                                     : halo_value(halo->east, 0, y, z, ny, nz);
                    if ((center <= isovalue && v > isovalue) || (center > isovalue && v <= isovalue))
                        isovalue_count++;
                }
                // -y neighbour (skip at global south boundary)
                if (gy - 1 >= 0) {
                    v = (y - 1 >= 0) ? field[index3d(x, y-1, z, ny, nz)]
                                     : halo_value(halo->south, radius - 1, x, z, nx, nz);
                    if ((center <= isovalue && v > isovalue) || (center > isovalue && v <= isovalue))
                        isovalue_count++;
                }
                // +y neighbour (skip at global north boundary)
                if (gy + 1 < gy_max) {
                    v = (y + 1 < ny) ? field[index3d(x, y+1, z, ny, nz)]
                                     : halo_value(halo->north, 0, x, z, nx, nz);
                    if ((center <= isovalue && v > isovalue) || (center > isovalue && v <= isovalue))
                        isovalue_count++;
                }
                // -z neighbour (skip at global back boundary)
                if (gz - 1 >= 0) {
                    v = (z - 1 >= 0) ? field[index3d(x, y, z-1, ny, nz)]
                                     : halo_value(halo->back, radius - 1, x, y, nx, ny);
                    if ((center <= isovalue && v > isovalue) || (center > isovalue && v <= isovalue))
                        isovalue_count++;
                }
                // +z neighbour (skip at global front boundary)
                if (gz + 1 < gz_max) {
                    v = (z + 1 < nz) ? field[index3d(x, y, z+1, ny, nz)]
                                     : halo_value(halo->front, 0, x, y, nx, ny);
                    if ((center <= isovalue && v > isovalue) || (center > isovalue && v <= isovalue))
                        isovalue_count++;
                }
            }
        }
    }

    return isovalue_count;
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
    long long *local_counts;
    long long *global_counts;
    int rank_x, rank_y, rank_z;
    int gx_off, gy_off, gz_off, gx_max, gy_max, gz_max;
    double start_time;
    double end_time;
    double local_time;
    double max_time;
    int radius;
    ProcessGrid grid;
    HaloBuffers halo = {0};

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
    radius = (d - 1) / 6;
    (void)ppn;  // ppn is accepted for interface compatibility but unused at runtime

    if (px * py * pz != comm_size) {
        if (rank == 0) {
            fprintf(stderr, "px * py * pz must equal the number of MPI processes.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // The stencil reaches `radius` cells in each direction; the local tile must
    // be at least that wide so no single step can skip over a tile entirely.
    if (d >= nx || d >= ny || d >= nz) {
        if (rank == 0) {
            fprintf(stderr, "d must be less than nx, ny, and nz.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Valid d values are 7, 13, 19, … (d = 6r + 1 for r = 1, 2, 3, …).
    if (d < 7 || (d - 1) % 6 != 0) {
        if (rank == 0) {
            fprintf(stderr, "d must be 7, 13, 19, ...\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    setup_process_grid(&grid, rank, px, py, pz);

    // All F fields share a single contiguous allocation; data[i] and next[i]
    // are set to sub-array pointers within data_storage / next_storage below.
    data = malloc((size_t)F * sizeof *data);
    next = malloc((size_t)F * sizeof *next);
    data_storage = malloc((size_t)F * arrSize * sizeof *data_storage);
    next_storage = malloc((size_t)F * arrSize * sizeof *next_storage);
    local_counts = calloc((size_t)F, sizeof *local_counts);
    global_counts = calloc((size_t)F, sizeof *global_counts);

    if (data == NULL || next == NULL || data_storage == NULL || next_storage == NULL ||
        local_counts == NULL || global_counts == NULL ||
        !allocate_halos(&halo, radius, nx, ny, nz)) {
        fprintf(stderr, "Rank %d could not allocate data array.\n", rank);
        free(data);
        free(next);
        free(data_storage);
        free(next_storage);
        free(local_counts);
        free(global_counts);
        free_halos(&halo);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    set_halo_neighbors(&halo, &grid);

    // Compute this process's (x, y, z) position in the process grid and its
    // corresponding offsets / extents in the global domain.
    rank_x = rank % px;
    rank_y = (rank / px) % py;
    rank_z = rank / (px * py);
    gx_off = rank_x * nx;
    gy_off = rank_y * ny;
    gz_off = rank_z * nz;
    gx_max = px * nx;
    gy_max = py * ny;
    gz_max = pz * nz;

    for (int i = 0; i < F; i++) {
        data[i] = data_storage + (size_t)i * arrSize;
        next[i] = next_storage + (size_t)i * arrSize;
    }

    // Initialise each field with pseudo-random values.  The loop enumerates
    // cells in x-fastest order (z slowest → y → x fastest) so the j-th
    // rand() call is assigned to the same logical (x,y,z) cell regardless
    // of how the flat index j is decoded.
    srand(seed);
    for (int i = 0; i < F; i++) {
        for (int j = 0; j < arrSize; j++) {
            int z_loc = j / (ny * nx);
            int rem   = j % (ny * nx);
            int y_loc = rem / nx;
            int x_loc = rem % nx;
            data[i][index3d(x_loc, y_loc, z_loc, ny, nz)] =
                (double)rand() * (rank + 1) / (110426.0 + i + j);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    for (int t = 0; t < T; t++) {
        for (int i = 0; i < F; i++) {
            exchange_halos(data[i], &grid, &halo);
            // Count isovalue crossings on the current field (with halos filled)
            // before applying the stencil.
            local_counts[i] = count_isovalue_cells(data[i], nx, ny, nz, isovalue,
                                                    &halo,
                                                    gx_off, gy_off, gz_off,
                                                    gx_max, gy_max, gz_max);
            d_point_stencil(data[i], next[i], nx, ny, nz, d, &halo);
        }

        // Sum per-field isovalue counts across all ranks; rank 0 prints results.
        MPI_Reduce(local_counts, global_counts, F, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int i = 0; i < F; i++) {
                printf("%lld", global_counts[i]);
                if (i < F - 1) printf(" ");
            }
            printf("\n");
        }

        // Swap current and next field pointers to avoid a copy.
        double **temp = data;
        data = next;
        next = temp;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    local_time = end_time - start_time;

    // Report the wall-clock time of the slowest rank (the true elapsed time).
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("%.6f\n", max_time);
    }

    free(global_counts);
    free(local_counts);
    free(next_storage);
    free(data_storage);
    free(next);
    free(data);
    free_halos(&halo);

    MPI_Finalize();
    return 0;
}
