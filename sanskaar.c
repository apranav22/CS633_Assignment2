#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define DIR_XM 0
#define DIR_XP 1
#define DIR_YM 2
#define DIR_YP 3
#define DIR_ZM 4
#define DIR_ZP 5

static inline int rank_from_coord(int x, int y, int z, int px, int py, int pz){
    if (x< 0 || x>= px || y <0 || y >= py || z<0 || z >= pz) return -1;
    return (z*py+y)* px+ x;
}

static inline int64_t idx4(int f,int z,int y, int x, int sz, int sy, int sx) {
    return (((((int64_t)f *sz +z)* sy+ y) * sx) +x);
}

static void pack_x(const double *a, double *buf, int x0, int k, int F, int nx, int ny, int nz, int sx, int sy, int sz){
    (void)nx;
    int64_t p = 0;
    for (int f = 0; f <F; f++){
        for (int z = 0; z < nz; z++){
            int zz = z + k;
            for (int y = 0; y< ny; y++){
                int yy = y+ k;
                for (int x=0; x < k; x++) {
                    int xx = x0 + x;
                    buf[p++] = a[idx4(f, zz, yy, xx, sz, sy, sx)];
                }
            }
        }
    }
}

static void unpack_x(double *a, const double *buf, int x0,int k,int F, int nx, int ny, int nz, int sx, int sy,int sz){
    (void)nx;
    int64_t p = 0;
    for (int f = 0; f < F; f++){
        for (int z = 0; z < nz; z++){
            int zz = z + k;
            for (int y = 0; y < ny; y++){  
                int yy = y + k;
                for (int x = 0; x < k; x++) {
                    int xx = x0 + x;
                    a[idx4(f, zz, yy, xx,sz,sy, sx)] = buf[p++];
                }
            }
        }
    }
}

static void pack_y(const double *a, double *buf, int y0, int k,int F, int nx, int ny, int nz,int sx,int sy,int sz){
    (void)ny;
    int64_t p = 0;
    for (int f =0; f <F; f++) {
        for (int z = 0; z < nz; ++z) {
            int zz = z + k;
            for (int y = 0; y < k; ++y) {
                int yy = y0 + y;
                for (int x = 0; x < nx; ++x) {
                    int xx = x + k;
                    buf[p++] = a[idx4(f, zz, yy, xx, sz, sy, sx)];
                }
            }
        }
    }
}

static void unpack_y(double *a, const double *buf, int y0, int k,int F, int nx, int ny, int nz,int sx, int sy, int sz){
    (void)ny;
    int64_t p = 0;
    for (int f = 0; f < F; f++) {
        for (int z = 0; z < nz; z++) {
            int zz = z + k;
            for (int y = 0; y < k; y++) {
                int yy = y0 + y;
                for (int x = 0; x < nx; x++) {
                    int xx = x + k;
                    a[idx4(f, zz, yy, xx, sz, sy, sx)] = buf[p++];
                }
            }
        }
    }
}

static void pack_z(const double *a, double *buf, int z0, int k,int F, int nx,int ny,int nz,int sx, int sy, int sz){
    (void)nz;
    int64_t p = 0;
    for (int f = 0; f < F; f++){
        for (int z = 0; z < k; z++) {
            int zz = z0 + z;
            for (int y = 0; y < ny; y++) {
                int yy = y + k;
                for (int x = 0; x < nx; x++) {
                    int xx = x + k;
                    buf[p++] = a[idx4(f, zz, yy, xx, sz, sy, sx)];
                }
            }
        }
    }
}

static void unpack_z(double *a, const double *buf, int z0, int k,int F, int nx, int ny,int nz,int sx, int sy, int sz){
    (void)nz;
    int64_t p = 0;
    for (int f = 0; f < F; f++) {
        for (int z = 0; z < k; z++) {
            int zz = z0 + z;
            for (int y = 0; y < ny; y++){
                int yy = y + k;
                for (int x = 0; x < nx; x++){
                    int xx = x + k;
                    a[idx4(f, zz, yy, xx, sz, sy, sx)] = buf[p++];
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 13) {
        if (myrank == 0) {
            fprintf(stderr,
                    "Usage: %s d ppn px py pz nx ny nz T seed F isovalue\n",
                    argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const int d = atoi(argv[1]);
    const int ppn = atoi(argv[2]);
    const int px = atoi(argv[3]);
    const int py = atoi(argv[4]);
    const int pz = atoi(argv[5]);
    const int nx = atoi(argv[6]);
    const int ny = atoi(argv[7]);
    const int nz = atoi(argv[8]);
    const int T = atoi(argv[9]);
    const int seed = atoi(argv[10]);
    const int F = atoi(argv[11]);
    const double isovalue = atof(argv[12]);
    (void)ppn;

    if (px*py*pz != nprocs) {
        if (myrank == 0) {
            fprintf(stderr, "Error: px*py*pz (%d) != #MPI processes (%d)\n", px * py * pz, nprocs);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (d<1 || (d-1) % 6 != 0) {
        if (myrank == 0) {
            fprintf(stderr, "Error: d must be of the form 1+6k (e.g., 7,13,19,...)\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    const int k = (d - 1) / 6;
    if (k >= nx || k >= ny || k >= nz) {
        if (myrank == 0) {
            fprintf(stderr, "Error: stencil radius k=(d-1)/6 must satisfy k<nx, k<ny, k<nz\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 4);
    }

    const int sx = nx+ 2* k;
    const int sy = ny+2 * k;
    const int sz = nz + 2 * k;
    const int64_t per_field = (int64_t)sx * sy * sz;
    const int64_t total = (int64_t)F * per_field;

    double *curr = (double *)calloc((size_t)total, sizeof(double));
    double *next = (double *)calloc((size_t)total,sizeof(double));
    if (!curr || !next) {
        fprintf(stderr, "Rank %d: memory allocation failed\n", myrank);
        MPI_Abort(MPI_COMM_WORLD, 5);
    }

    const int cx = myrank % px;
    const int cy = (myrank/ px) % py;
    const int cz = myrank/ (px * py);

    int nbr[6];
    nbr[DIR_XM] = rank_from_coord(cx - 1, cy, cz, px, py, pz);
    nbr[DIR_XP] = rank_from_coord(cx + 1, cy, cz, px, py, pz);
    nbr[DIR_YM] = rank_from_coord(cx, cy - 1, cz, px, py, pz);
    nbr[DIR_YP] = rank_from_coord(cx, cy + 1, cz, px, py, pz);
    nbr[DIR_ZM] = rank_from_coord(cx, cy, cz - 1, px, py, pz);
    nbr[DIR_ZP] = rank_from_coord(cx, cy, cz + 1, px, py, pz);

    const int64_t halo_sz[6] = {
        (int64_t)F * k * ny * nz,
        (int64_t)F * k * ny * nz,
        (int64_t)F * nx * k * nz,
        (int64_t)F * nx * k * nz,
        (int64_t)F * nx * ny * k,
        (int64_t)F * nx * ny * k
    };

    double *sbuf[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
    double *rbuf[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
    for (int dir = 0; dir < 6; ++dir) {
        if (nbr[dir] != -1) {
            sbuf[dir] = (double *)malloc((size_t)halo_sz[dir] *sizeof(double));
            rbuf[dir] = (double *)malloc((size_t)halo_sz[dir] *sizeof(double));
            if (!sbuf[dir] || !rbuf[dir]) {
                fprintf(stderr, "Rank %d: halo buffer allocation failed\n", myrank);
                MPI_Abort(MPI_COMM_WORLD, 6);
            }
        }
    }

    srand(seed);
    const int64_t arrSize = (int64_t)nx * ny * nz;
    for (int f = 0; f < F; ++f) {
        for (int64_t j = 0; j < arrSize; ++j) {
            int64_t z = j / ((int64_t)ny* nx);
            int64_t rem = j % ((int64_t)ny* nx);
            int64_t y = rem / nx;
            int64_t x = rem % nx;
            int xx = (int)x + k;
            int yy = (int)y + k;
            int zz = (int)z + k;
            curr[idx4(f, zz, yy, xx, sz, sy, sx)] =
                ((double)rand() * (myrank + 1)) / (110426.0 + f + (double)j);
        }
    }

    int64_t *local_counts = (int64_t *)malloc((size_t)F * sizeof(int64_t));
    int64_t *global_counts = (int64_t *)malloc((size_t)F * sizeof(int64_t));
    if (!local_counts || !global_counts) {
        fprintf(stderr, "Rank %d: count array allocation failed\n", myrank);
        MPI_Abort(MPI_COMM_WORLD, 7);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int t = 0; t < T; ++t) {
        MPI_Request req[12];
        int rq = 0;

        if (nbr[DIR_XM] != -1) {
            MPI_Irecv(rbuf[DIR_XM], (int)halo_sz[DIR_XM], MPI_DOUBLE, nbr[DIR_XM], 0, MPI_COMM_WORLD, &req[rq++]);
        }
        if (nbr[DIR_XP] != -1) {
            MPI_Irecv(rbuf[DIR_XP], (int)halo_sz[DIR_XP], MPI_DOUBLE, nbr[DIR_XP], 1, MPI_COMM_WORLD, &req[rq++]);
        }
        if (nbr[DIR_YM] != -1) {
            MPI_Irecv(rbuf[DIR_YM], (int)halo_sz[DIR_YM], MPI_DOUBLE, nbr[DIR_YM], 2, MPI_COMM_WORLD, &req[rq++]);
        }
        if (nbr[DIR_YP] != -1) {
            MPI_Irecv(rbuf[DIR_YP], (int)halo_sz[DIR_YP], MPI_DOUBLE, nbr[DIR_YP], 3, MPI_COMM_WORLD, &req[rq++]);
        }
        if (nbr[DIR_ZM] != -1) {
            MPI_Irecv(rbuf[DIR_ZM], (int)halo_sz[DIR_ZM], MPI_DOUBLE, nbr[DIR_ZM], 4, MPI_COMM_WORLD, &req[rq++]);
        }
        if (nbr[DIR_ZP] != -1) {
            MPI_Irecv(rbuf[DIR_ZP], (int)halo_sz[DIR_ZP], MPI_DOUBLE, nbr[DIR_ZP], 5, MPI_COMM_WORLD, &req[rq++]);
        }

        if (nbr[DIR_XM] != -1) {
            pack_x(curr, sbuf[DIR_XM], k, k, F, nx, ny, nz, sx, sy, sz);
            MPI_Isend(sbuf[DIR_XM], (int)halo_sz[DIR_XM], MPI_DOUBLE, nbr[DIR_XM], 1, MPI_COMM_WORLD, &req[rq++]);
        }
        if (nbr[DIR_XP] != -1) {
            pack_x(curr, sbuf[DIR_XP], nx, k, F, nx, ny, nz, sx, sy, sz);
            MPI_Isend(sbuf[DIR_XP], (int)halo_sz[DIR_XP], MPI_DOUBLE, nbr[DIR_XP], 0, MPI_COMM_WORLD, &req[rq++]);
        }
        if (nbr[DIR_YM] != -1) {
            pack_y(curr, sbuf[DIR_YM], k, k, F, nx, ny, nz, sx, sy, sz);
            MPI_Isend(sbuf[DIR_YM], (int)halo_sz[DIR_YM], MPI_DOUBLE, nbr[DIR_YM], 3, MPI_COMM_WORLD, &req[rq++]);
        }
        if (nbr[DIR_YP] != -1) {
            pack_y(curr, sbuf[DIR_YP], ny, k, F, nx, ny, nz, sx, sy, sz);
            MPI_Isend(sbuf[DIR_YP], (int)halo_sz[DIR_YP], MPI_DOUBLE, nbr[DIR_YP], 2, MPI_COMM_WORLD, &req[rq++]);
        }
        if (nbr[DIR_ZM] != -1) {
            pack_z(curr, sbuf[DIR_ZM], k, k, F, nx, ny, nz, sx, sy, sz);
            MPI_Isend(sbuf[DIR_ZM], (int)halo_sz[DIR_ZM], MPI_DOUBLE, nbr[DIR_ZM], 5, MPI_COMM_WORLD, &req[rq++]);
        }
        if (nbr[DIR_ZP] != -1) {
            pack_z(curr, sbuf[DIR_ZP], nz, k, F, nx, ny, nz, sx, sy, sz);
            MPI_Isend(sbuf[DIR_ZP], (int)halo_sz[DIR_ZP], MPI_DOUBLE, nbr[DIR_ZP], 4, MPI_COMM_WORLD, &req[rq++]);
        }

        if (rq > 0) {
            MPI_Waitall(rq, req, MPI_STATUSES_IGNORE);
        }

        if (nbr[DIR_XM] != -1) unpack_x(curr, rbuf[DIR_XM], 0, k, F, nx, ny, nz, sx, sy, sz);
        if (nbr[DIR_XP] != -1) unpack_x(curr, rbuf[DIR_XP], nx + k, k, F, nx, ny, nz, sx, sy, sz);
        if (nbr[DIR_YM] != -1) unpack_y(curr, rbuf[DIR_YM], 0, k, F, nx, ny, nz, sx, sy, sz);
        if (nbr[DIR_YP] != -1) unpack_y(curr, rbuf[DIR_YP], ny + k, k, F, nx, ny, nz, sx, sy, sz);
        if (nbr[DIR_ZM] != -1) unpack_z(curr, rbuf[DIR_ZM], 0, k, F, nx, ny, nz, sx, sy, sz);
        if (nbr[DIR_ZP] != -1) unpack_z(curr, rbuf[DIR_ZP], nz + k, k, F, nx, ny, nz, sx, sy, sz);

        for (int f = 0; f < F; ++f) local_counts[f] = 0;

        for (int f = 0; f < F; ++f) {
            for (int z = 0; z < nz; ++z) {
                int gz = cz * nz + z;
                int zz = z + k;
                for (int y = 0; y < ny; ++y) {
                    int gy = cy * ny + y;
                    int yy = y + k;
                    for (int x = 0; x < nx; ++x) {
                        int gx = cx * nx + x;
                        int xx = x + k;
                        double center = curr[idx4(f, zz, yy, xx, sz, sy, sx)];
                        double sum = center;
                        int m = 1;

                        for (int s = 1; s <= k; ++s) {
                            if (gx - s >= 0) {
                                sum += curr[idx4(f, zz, yy, xx - s, sz, sy, sx)];
                                ++m;
                            }
                            if (gx + s < px * nx) {
                                sum += curr[idx4(f, zz, yy, xx + s, sz, sy, sx)];
                                ++m;
                            }
                            if (gy - s >= 0) {
                                sum += curr[idx4(f, zz, yy - s, xx, sz, sy, sx)];
                                ++m;
                            }
                            if (gy + s < py * ny) {
                                sum += curr[idx4(f, zz, yy + s, xx, sz, sy, sx)];
                                ++m;
                            }
                            if (gz - s >= 0) {
                                sum += curr[idx4(f, zz - s, yy, xx, sz, sy, sx)];
                                ++m;
                            }
                            if (gz + s < pz * nz) {
                                sum += curr[idx4(f, zz + s, yy, xx, sz, sy, sx)];
                                ++m;
                            }
                        }

                        next[idx4(f, zz, yy, xx, sz, sy, sx)] = sum / (double)m;

                        if (gx - 1 >= 0) {
                            double v = curr[idx4(f, zz, yy, xx - 1, sz, sy, sx)];
                            if ((center <= isovalue && v > isovalue) || (center > isovalue && v <= isovalue)) local_counts[f]++;
                        }
                        if (gx + 1 < px * nx) {
                            double v = curr[idx4(f, zz, yy, xx + 1, sz, sy, sx)];
                            if ((center <= isovalue && v > isovalue) || (center > isovalue && v <= isovalue)) local_counts[f]++;
                        }
                        if (gy - 1 >= 0) {
                            double v = curr[idx4(f, zz, yy - 1, xx, sz, sy, sx)];
                            if ((center <= isovalue && v > isovalue) || (center > isovalue && v <= isovalue)) local_counts[f]++;
                        }
                        if (gy + 1 < py * ny) {
                            double v = curr[idx4(f, zz, yy + 1, xx, sz, sy, sx)];
                            if ((center <= isovalue && v > isovalue) || (center > isovalue && v <= isovalue)) local_counts[f]++;
                        }
                        if (gz - 1 >= 0) {
                            double v = curr[idx4(f, zz - 1, yy, xx, sz, sy, sx)];
                            if ((center <= isovalue && v > isovalue) || (center > isovalue && v <= isovalue)) local_counts[f]++;
                        }
                        if (gz + 1 < pz * nz) {
                            double v = curr[idx4(f, zz + 1, yy, xx, sz, sy, sx)];
                            if ((center <= isovalue && v > isovalue) || (center > isovalue && v <= isovalue)) local_counts[f]++;
                        }
                    }
                }
            }
        }

        MPI_Reduce(local_counts, global_counts, F, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (myrank == 0) {
            for (int f =0; f < F; ++f){
                if (f) putchar(' ');
                printf("%lld", (long long)global_counts[f]);
            }
            putchar('\n');
        }

        double *tmp = curr;
        curr = next;
        next = tmp;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (myrank == 0) {
        printf("%.6f\n", t1- t0);
    }

    for (int dir = 0; dir < 6; dir++) {
        free(sbuf[dir]);
        free(rbuf[dir]);
    }
    free(local_counts);
    free(global_counts);
    free(curr);
    free(next);

    MPI_Finalize();
    return 0;
}