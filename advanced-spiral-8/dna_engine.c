/*
 * ═══════════════════════════════════════════════════════════════════════════
 * DNA ENGINE - Real-time φ-Spiral Stream Generator
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * PURPOSE: High-performance C backend for ecoli46.py cell division visualization
 *          Provides streaming spiral generation with unified framework integration
 *
 * ARCHITECTURE:
 *   - Shared memory IPC (named pipe/socket)
 *   - Frame-by-frame streaming (400 points/frame)
 *   - Multi-cell support with division tracking
 *   - Geometry state caching for organelle physics
 *
 * COMPILATION:
 *   Windows: tcc -o dna_engine.exe dna_engine.c -shared -rdynamic
 *   Linux:   gcc -o dna_engine.so dna_engine.c -shared -fPIC -lm -O3
 *
 * PYTHON INTERFACE:
 *   import ctypes
 *   engine = ctypes.CDLL('./dna_engine.so')
 *   engine.init_engine(b'ecoli_k12.fasta')
 *   pts = engine.get_frame(frame_num, cell_id)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(_WIN32) || defined(__TINYC__)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS (Unified Framework)
// ═══════════════════════════════════════════════════════════════════════════

#define PHI 1.618033988749895
#define PI 3.141592653589793
#define SQRT5 2.23606797749979
#define PHI_INV 0.6180339887498948482
#define GOLDEN_ANGLE_DEG (360.0 / (PHI * PHI))  // 137.507764°
#define GOLDEN_ANGLE_RAD (GOLDEN_ANGLE_DEG * PI / 180.0)

#define MAX_CELLS 128
#define POINTS_PER_FRAME 400
#define MAX_GENOME_SIZE (8 * 1024 * 1024)  // 8MB
#define CORE_RADIUS 15.0
#define STRAND_SEP 0.5
#define TWIST_FACTOR (2.0 * PI)

// DNA Base mapping
#define BASE_A 0
#define BASE_T 1
#define BASE_G 2
#define BASE_C 3
#define BASE_N 4
#define BASE_R 5
#define BASE_Y 6
#define BASE_W 7

// Geometry octave (matching ecoli46.py)
typedef struct {
    int dimension;
    const char* note;
    float color_r, color_g, color_b;
    const char* name;
    float alpha;
    int vertices;
} Geometry;

static const Geometry GEOMETRIES[8] = {
    {1, "C", 1.0f, 0.0f, 0.0f, "Point",       0.015269f,  1},
    {2, "D", 0.0f, 1.0f, 0.0f, "Line",        0.008262f,  2},
    {3, "E", 0.93f, 0.51f, 0.93f, "Triangle", 0.110649f,  3},
    {4, "F", 0.58f, 0.44f, 0.86f, "Tetrahedron", -0.083485f, 4},
    {5, "G", 0.0f, 0.0f, 1.0f, "Pentachoron", 0.025847f,  5},
    {6, "A", 0.29f, 0.0f, 0.51f, "Hexacross", -0.045123f, 12},
    {7, "B", 0.5f, 0.0f, 0.5f, "Heptacube",   0.067891f, 14},
    {8, "C", 1.0f, 1.0f, 1.0f, "Octacube",    0.012345f, 16},
};

// Fibonacci/Prime tables for D_n operator
static const double FIB_CACHE[93] = {
    1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987,
    1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393,
    196418, 317811, 514229, 832040, 1346269, 2178309, 3524578, 5702887,
    9227465, 14930352, 24157817, 39088169, 63245986, 102334155, 165580141,
    267914296, 433494437, 701408733, 1134903170, 1836311903, 2971215073,
    4807526976, 7778742049, 12586269025, 20365011074, 32951280099,
    53316291173, 86267571272, 139583862445, 225851433717, 365435296162,
    591286729879, 956722026041, 1548008755920, 2504730781961,
    4052739537881, 6557470319842, 10610209857723, 17167680177565,
    27777890035288, 44945570212853, 72723460248141, 117669030460994,
    190392490709135, 308061521170129, 498454011879264, 806515533049393,
    1304969544928657, 2111485077978050, 3416454622906707, 5527939700884757,
    8944394323791464, 14472334024676221, 23416728348467685,
    37889062373143906, 61305790721611591, 99194853094755497,
    160500643816367088, 259695496911122585, 420196140727489673,
    679891637638612258, 1100087778366101931, 1779979416004714189,
    2880067194370816120, 4660046610375530309, 7540113804746346429
};

static const int PRIMES_CACHE[168] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
    71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139,
    149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
    227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293,
    307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383,
    389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
    467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569,
    571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647,
    653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743,
    751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839,
    853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
    947, 953, 967, 971, 977, 983, 991, 997
};

// ═══════════════════════════════════════════════════════════════════════════
// GLOBAL STATE
// ═══════════════════════════════════════════════════════════════════════════

typedef struct {
    char* sequence;
    size_t length;
    char* name;
} Genome;

typedef struct {
    int id;
    int frame;
    double center_offset[3];
    int active;
} Cell;

static Genome g_genome = {0};
static Cell g_cells[MAX_CELLS] = {0};
static int g_num_cells = 0;
static double g_angles[8];  // Cached geometry angles

// ═══════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

static int base_to_index(char base) {
    switch(base) {
        case 'A': return BASE_A;
        case 'T': return BASE_T;
        case 'G': return BASE_G;
        case 'C': return BASE_C;
        case 'N': return BASE_N;
        case 'R': return BASE_R;
        case 'Y': return BASE_Y;
        case 'W': return BASE_W;
        default: return BASE_A;
    }
}

static int base_to_geometry(char base) {
    // Map from ecoli46.py: base_map = {'A': 5, 'T': 2, 'G': 4, 'C': 1}
    switch(base) {
        case 'A': return 4;  // Pentachoron (5D) -> index 4
        case 'T': return 1;  // Line (2D) -> index 1
        case 'G': return 3;  // Tetrahedron (4D) -> index 3
        case 'C': return 0;  // Point (1D) -> index 0
        default: return 0;
    }
}

static double get_fib(int n) {
    if (n < 0 || n >= 93) return 1.0;
    return FIB_CACHE[n];
}

static double get_prime(int n) {
    if (n < 0 || n >= 168) return 2.0;
    return (double)PRIMES_CACHE[n];
}

// D_n operator (simplified for real-time streaming)
static double D_n(int n, double r) {
    if (n < 0) n = 0;
    double F_n = get_fib(n % 93);
    double P_n = get_prime(n % 168);
    double base_n = pow(2.0, n % 12);  // Dyadic scaling
    return sqrt(PHI * F_n * base_n * P_n) * r;
}

// ═══════════════════════════════════════════════════════════════════════════
// GENOME LOADING
// ═══════════════════════════════════════════════════════════════════════════

EXPORT int init_engine(const char* fasta_path) {
    FILE* fp = fopen(fasta_path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s\n", fasta_path);
        return -1;
    }

    // Allocate sequence buffer
    g_genome.sequence = malloc(MAX_GENOME_SIZE);
    if (!g_genome.sequence) {
        fclose(fp);
        return -1;
    }

    g_genome.length = 0;
    char line[4096];
    int in_sequence = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '>') {
            if (!g_genome.name) {
                size_t len = strlen(line);
                g_genome.name = malloc(len);
                if (g_genome.name) {
                    strncpy(g_genome.name, line + 1, len - 2);
                    g_genome.name[len - 2] = '\0';
                }
            }
            in_sequence = 1;
            continue;
        }

        if (in_sequence) {
            for (size_t i = 0; line[i] != '\0' && line[i] != '\n'; i++) {
                char c = line[i];
                if (c >= 'a' && c <= 'z') c = c - 'a' + 'A';
                if ((c >= 'A' && c <= 'Z') || c == 'N') {
                    if (g_genome.length < MAX_GENOME_SIZE - 1) {
                        g_genome.sequence[g_genome.length++] = c;
                    }
                }
            }
        }
    }

    fclose(fp);
    g_genome.sequence[g_genome.length] = '\0';

    // Initialize geometry angles (matching ecoli46.py)
    for (int i = 0; i < 8; i++) {
        g_angles[i] = i * GOLDEN_ANGLE_DEG * PI / 180.0;
    }

    // Initialize first cell
    g_cells[0].id = 0;
    g_cells[0].frame = 0;
    g_cells[0].center_offset[0] = 0.0;
    g_cells[0].center_offset[1] = 0.0;
    g_cells[0].center_offset[2] = 0.0;
    g_cells[0].active = 1;
    g_num_cells = 1;

    printf("DNA Engine initialized: %zu bases\n", g_genome.length);
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// FRAME GENERATION (Streaming Interface)
// ═══════════════════════════════════════════════════════════════════════════

typedef struct {
    float x, y, z;
    float color_r, color_g, color_b;
    int dimension;
    char base;
} Point;

EXPORT int get_frame_data(int cell_id, int frame_num, Point* strand1_out, Point* strand2_out) {
    if (cell_id >= MAX_CELLS || !g_cells[cell_id].active) {
        return -1;
    }

    Cell* cell = &g_cells[cell_id];
    double* offset = cell->center_offset;

    // Generate N points for this frame
    int N = POINTS_PER_FRAME;
    double t_start = frame_num;
    double t_end = frame_num + N;

    for (int i = 0; i < N; i++) {
        double tt = t_start + i;
        int idx = (int)tt % g_genome.length;
        char base = g_genome.sequence[idx];
        int geom_idx = base_to_geometry(base);

        const Geometry* geom = &GEOMETRIES[geom_idx];

        // Spiral parameters (matching ecoli46.py)
        double progress = tt / (double)g_genome.length;
        double r = CORE_RADIUS * (1.0 - pow(progress, 1.5));
        if (r < 0.5) r = 0.5;

        double theta = GOLDEN_ANGLE_RAD * tt;
        double twist = progress * TWIST_FACTOR;
        double z = sin(progress * PI * 4.0) * 2.0 + progress * 8.0;

        // Strand 1 (positive rotation)
        double a1 = g_angles[geom_idx];
        double x1 = r * cos(theta) * cos(a1) - r * sin(theta) * sin(a1);
        double y1 = r * sin(theta) * cos(a1) + r * cos(theta) * sin(a1);

        strand1_out[i].x = (float)(x1 + offset[0]);
        strand1_out[i].y = (float)(y1 + offset[1]);
        strand1_out[i].z = (float)(z + offset[2]);
        strand1_out[i].color_r = geom->color_r;
        strand1_out[i].color_g = geom->color_g;
        strand1_out[i].color_b = geom->color_b;
        strand1_out[i].dimension = geom->dimension;
        strand1_out[i].base = base;

        // Strand 2 (negative rotation)
        double a2 = -g_angles[geom_idx];
        double x2 = r * cos(theta) * cos(a2) - r * sin(theta) * sin(a2) + STRAND_SEP;
        double y2 = r * sin(theta) * cos(a2) + r * cos(theta) * sin(a2) - STRAND_SEP;

        strand2_out[i].x = (float)(x2 + offset[0]);
        strand2_out[i].y = (float)(y2 + offset[1]);
        strand2_out[i].z = (float)(z + offset[2]);
        strand2_out[i].color_r = geom->color_r;
        strand2_out[i].color_g = geom->color_g;
        strand2_out[i].color_b = geom->color_b;
        strand2_out[i].dimension = geom->dimension;
        strand2_out[i].base = base;
    }

    return N;
}

// ═══════════════════════════════════════════════════════════════════════════
// CELL DIVISION
// ═══════════════════════════════════════════════════════════════════════════

EXPORT int create_daughter_cell(int parent_id, double offset_x, double offset_y, double offset_z) {
    if (g_num_cells >= MAX_CELLS) {
        return -1;  // Maximum cells reached
    }

    Cell* parent = &g_cells[parent_id];
    int new_id = g_num_cells++;

    g_cells[new_id].id = new_id;
    g_cells[new_id].frame = 0;
    g_cells[new_id].center_offset[0] = parent->center_offset[0] + offset_x;
    g_cells[new_id].center_offset[1] = parent->center_offset[1] + offset_y;
    g_cells[new_id].center_offset[2] = parent->center_offset[2] + offset_z;
    g_cells[new_id].active = 1;

    return new_id;
}

// ═══════════════════════════════════════════════════════════════════════════
// QUERY INTERFACE
// ═══════════════════════════════════════════════════════════════════════════

EXPORT int get_genome_length() {
    return (int)g_genome.length;
}

EXPORT int get_num_cells() {
    return g_num_cells;
}

EXPORT const char* get_genome_name() {
    return g_genome.name ? g_genome.name : "Unknown";
}

EXPORT void cleanup_engine() {
    if (g_genome.sequence) {
        free(g_genome.sequence);
        g_genome.sequence = NULL;
    }
    if (g_genome.name) {
        free(g_genome.name);
        g_genome.name = NULL;
    }
    g_genome.length = 0;
    g_num_cells = 0;
}
