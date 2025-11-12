/*
 * ═══════════════════════════════════════════════════════════════════════════
 * FASTA DNA UNIFIED FRAMEWORK V1 - GENOME-DRIVEN φ-SPIRAL EVOLUTION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * PURPOSE: Native C port of fasta4.py with unified BigG+Fudge10 framework
 *          integration for accelerated φ-spiral DNA visualization
 *
 * FRAMEWORK INTEGRATION:
 *   - Unified D_n operator from BigG/Fudge10
 *   - 8 geometric octaves mapped to DNA bases
 *   - φ-harmonic coupling with 4096-bit APA backend
 *   - Genome-driven recursive emergence
 *
 * COMPILATION:
 *   gcc -o fasta_dna_unified_v1 fasta_dna_unified_v1.c -lm -O3 -march=native
 *
 * DEPENDENCIES:
 *   - ecoli_k12.fasta (E. coli K-12 genome)
 *   - Standard C math library
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ═══════════════════════════════════════════════════════════════════════════
// COMPATIBILITY HELPERS
// ═══════════════════════════════════════════════════════════════════════════

// strndup implementation for Windows/TinyCC
#if defined(_WIN32) || defined(__TINYC__)
static char* strndup(const char* s, size_t n) {
    size_t len = strlen(s);
    if (len > n) len = n;
    char* result = malloc(len + 1);
    if (result) {
        memcpy(result, s, len);
        result[len] = '\0';
    }
    return result;
}
#endif

// ═══════════════════════════════════════════════════════════════════════════
// FUNDAMENTAL CONSTANTS (Unified Framework)
// ═══════════════════════════════════════════════════════════════════════════

#define PHI 1.618033988749895        // Golden ratio
#define PI 3.141592653589793
#define SQRT5 2.23606797749979
#define PHI_INV 0.6180339887498948482 // 1/φ
#define GOLDEN_ANGLE_DEG (360.0 / (PHI * PHI))  // 137.507764°

// DNA Base to Dimension mapping (8 geometries)
#define BASE_A 0  // Point (1D)
#define BASE_T 1  // Line (2D)
#define BASE_G 2  // Triangle (3D)
#define BASE_C 3  // Tetrahedron (4D)

// Extended bases for 8-geometry framework
#define BASE_N 4  // Pentachoron (5D)
#define BASE_R 5  // Hexacross (6D)
#define BASE_Y 6  // Heptacube (7D)
#define BASE_W 7  // Octacube (8D)

// Primes for D_n operator
static const int PRIMES[50] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229
};

// Fibonacci cache (precomputed for speed)
static double FIB_CACHE[100];

// 8 Geometries (from Spiral8 framework)
typedef struct {
    int dimension;
    const char *note;
    const char *color;
    const char *geometry;
    double alpha;     // Growth exponent
    int vertices;
    double phi_factor;
} Geometry;

static const Geometry GEOMETRIES[8] = {
    {1, "C", "red",          "Point",        0.015269, 1,  0.064681},
    {2, "D", "green",        "Line",         0.008262, 2,  0.021630},
    {3, "E", "violet",       "Triangle",     0.110649, 3,  0.179034},
    {4, "F", "mediumpurple", "Tetrahedron", -0.083485, 4, -0.083485},
    {5, "G", "blue",         "Pentachoron",  0.025847, 5,  0.015974},
    {6, "A", "indigo",       "Hexacross",   -0.045123, 12, -0.017235},
    {7, "B", "purple",       "Heptacube",    0.067891, 14,  0.016027},
    {8, "C", "white",        "Octacube",     0.012345, 16,  0.001801}
};

// ═══════════════════════════════════════════════════════════════════════════
// GENOME STRUCTURE
// ═══════════════════════════════════════════════════════════════════════════

typedef struct {
    char *sequence;
    size_t length;
    const char *name;
} Genome;

// ═══════════════════════════════════════════════════════════════════════════
// φ-SPIRAL STATE
// ═══════════════════════════════════════════════════════════════════════════

typedef struct {
    double x, y, z;
    double phase;
    double amplitude;
    int dimension;
    char base;
} SpiralPoint;

typedef struct {
    SpiralPoint *strand1;
    SpiralPoint *strand2;
    size_t capacity;
    size_t count;
} SpiralState;

// ═══════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

void init_fibonacci_cache() {
    // Precompute Fibonacci numbers using Binet's formula
    for (int i = 0; i < 100; i++) {
        double term1 = pow(PHI, i) / SQRT5;
        double term2 = pow(PHI_INV, i) * cos(PI * i);
        FIB_CACHE[i] = term1 - term2;
    }
}

double fibonacci_real(double n) {
    if (n >= 0 && n < 100 && n == floor(n)) {
        return FIB_CACHE[(int)n];
    }
    // For non-integer or out of range, use Binet's formula
    if (n > 100) return 0.0;  // Avoid overflow
    double term1 = pow(PHI, n) / SQRT5;
    double term2 = pow(PHI_INV, n) * cos(PI * n);
    return term1 - term2;
}

double prime_product_index(double n, double beta) {
    int idx = ((int)floor(n + beta) + 50) % 50;
    return (double)PRIMES[idx];
}

// ═══════════════════════════════════════════════════════════════════════════
// UNIFIED D_n OPERATOR
// ═══════════════════════════════════════════════════════════════════════════

double D_n(double n, double beta, double r, double k, double Omega, double base) {
    /*
     * Universal D_n operator:
     * D_n(r) = sqrt(phi * F_n * base^n * P_n * Omega) * r^k
     */
    double Fn = fibonacci_real(n + beta);
    double Pn = prime_product_index(n, beta);
    double dyadic = pow(base, n + beta);

    double val = PHI * fabs(Fn) * dyadic * Pn * Omega;
    val = fmax(val, 1e-15);  // Prevent underflow

    double result = sqrt(val) * pow(r, k);

    // Apply Fibonacci sign
    if (Fn < 0) result = -result;

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// GENOME LOADING
// ═══════════════════════════════════════════════════════════════════════════

int base_to_dimension(char base) {
    switch (base) {
        case 'A': return BASE_A;
        case 'T': return BASE_T;
        case 'G': return BASE_G;
        case 'C': return BASE_C;
        case 'N': return BASE_N;
        case 'R': return BASE_R;
        case 'Y': return BASE_Y;
        case 'W': return BASE_W;
        default:  return BASE_A;  // Default to point geometry
    }
}

Genome* load_genome(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open FASTA file: %s\n", filename);
        return NULL;
    }

    Genome *genome = malloc(sizeof(Genome));
    if (!genome) {
        fclose(fp);
        return NULL;
    }

    // Allocate initial buffer
    size_t capacity = 1024 * 1024;  // 1MB initial
    genome->sequence = malloc(capacity);
    genome->length = 0;
    genome->name = NULL;

    char line[4096];
    int in_sequence = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '>') {
            // Header line
            if (!genome->name) {
                size_t len = strlen(line);
                genome->name = strndup(line + 1, len - 2);  // Skip '>' and newline
            }
            in_sequence = 1;
            continue;
        }

        if (in_sequence) {
            // Sequence line
            size_t len = strlen(line);

            // Reallocate if needed
            while (genome->length + len >= capacity) {
                capacity *= 2;
                genome->sequence = realloc(genome->sequence, capacity);
                if (!genome->sequence) {
                    fprintf(stderr, "Error: Memory allocation failed\n");
                    free(genome);
                    fclose(fp);
                    return NULL;
                }
            }

            // Copy bases (convert to uppercase, skip whitespace)
            for (size_t i = 0; i < len; i++) {
                char c = line[i];
                if (c >= 'a' && c <= 'z') c = c - 'a' + 'A';
                if (c >= 'A' && c <= 'Z') {
                    genome->sequence[genome->length++] = c;
                }
            }
        }
    }

    fclose(fp);
    genome->sequence[genome->length] = '\0';

    printf("Loaded genome: %s\n", genome->name ? genome->name : "Unknown");
    printf("Length: %zu bases\n", genome->length);

    return genome;
}

void free_genome(Genome *genome) {
    if (genome) {
        if (genome->sequence) free(genome->sequence);
        if (genome->name) free((void*)genome->name);
        free(genome);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// φ-SPIRAL COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════

SpiralState* spiral_init(size_t capacity) {
    SpiralState *state = malloc(sizeof(SpiralState));
    if (!state) return NULL;

    state->capacity = capacity;
    state->count = 0;
    state->strand1 = malloc(capacity * sizeof(SpiralPoint));
    state->strand2 = malloc(capacity * sizeof(SpiralPoint));

    if (!state->strand1 || !state->strand2) {
        free(state);
        return NULL;
    }

    return state;
}

void spiral_free(SpiralState *state) {
    if (state) {
        if (state->strand1) free(state->strand1);
        if (state->strand2) free(state->strand2);
        free(state);
    }
}

void compute_spiral_point(SpiralPoint *point, size_t idx, char base,
                          const Genome *genome, double period, double core_radius) {
    // Map base to dimension
    int dim = base_to_dimension(base);
    const Geometry *geom = &GEOMETRIES[dim];

    // Golden angle spiral
    double theta = idx * GOLDEN_ANGLE_DEG * PI / 180.0;

    // Vertical evolution (helical)
    double progress = (double)idx / genome->length;
    double z = sin(progress * PI * 4.0) * 2.0 + (progress * 8.0);

    // Radial decay (converging to center)
    double r = core_radius * (1.0 - pow(progress, 1.5));
    r = fmax(r, 0.5);  // Minimum radius

    // Rotation offset from geometry
    double angle_offset = dim * GOLDEN_ANGLE_DEG * PI / 180.0;

    // Apply D_n operator for amplitude modulation
    double n = (double)dim;
    double beta = progress;  // Use genome progress as beta
    double Omega = 1.0 + geom->alpha;
    double base_val = 2.0;
    double k = geom->phi_factor;

    double amplitude = D_n(n, beta, r, k, Omega, base_val);

    // Genome-derived noise (deterministic)
    double noise_x = (genome->sequence[(idx + 0) % genome->length] % 10) * 0.001 * (dim + 1);
    double noise_y = (genome->sequence[(idx + 1) % genome->length] % 10) * 0.001 * (dim + 1);
    double noise_z = (genome->sequence[(idx + 2) % genome->length] % 10) * 0.001 * (dim + 1);

    // Compute position
    point->x = r * cos(theta + angle_offset) + noise_x;
    point->y = r * sin(theta + angle_offset) + noise_y;
    point->z = z + noise_z;

    point->phase = theta;
    point->amplitude = amplitude;
    point->dimension = dim;
    point->base = base;
}

void generate_spiral(SpiralState *state, const Genome *genome, size_t max_points) {
    const double period = 13.057;
    const double core_radius = 15.0;
    const double strand_separation = 0.5;

    size_t num_points = (max_points < genome->length) ? max_points : genome->length;

    for (size_t i = 0; i < num_points; i++) {
        char base = genome->sequence[i];

        // Strand 1 (forward rotation)
        compute_spiral_point(&state->strand1[i], i, base, genome, period, core_radius);

        // Strand 2 (counter-rotation with separation)
        compute_spiral_point(&state->strand2[i], i, base, genome, period, core_radius);

        // Apply counter-rotation and separation to strand 2
        int dim = base_to_dimension(base);
        double angle_offset = -dim * GOLDEN_ANGLE_DEG * PI / 180.0;  // Negative for counter

        double theta = i * GOLDEN_ANGLE_DEG * PI / 180.0;
        double r = core_radius * (1.0 - pow((double)i / genome->length, 1.5));
        r = fmax(r, 0.5);

        state->strand2[i].x = r * cos(theta + angle_offset) + strand_separation;
        state->strand2[i].y = r * sin(theta + angle_offset) - strand_separation;
    }

    state->count = num_points;
}

// ═══════════════════════════════════════════════════════════════════════════
// OUTPUT & VISUALIZATION
// ═══════════════════════════════════════════════════════════════════════════

void export_to_csv(const SpiralState *state, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot create output file: %s\n", filename);
        return;
    }

    fprintf(fp, "strand,index,x,y,z,phase,amplitude,dimension,base,geometry\n");

    for (size_t i = 0; i < state->count; i++) {
        const SpiralPoint *p1 = &state->strand1[i];
        const SpiralPoint *p2 = &state->strand2[i];

        const char *geom_name = GEOMETRIES[p1->dimension].geometry;

        fprintf(fp, "1,%zu,%.6f,%.6f,%.6f,%.6f,%.6e,%d,%c,%s\n",
                i, p1->x, p1->y, p1->z, p1->phase, p1->amplitude,
                p1->dimension, p1->base, geom_name);

        fprintf(fp, "2,%zu,%.6f,%.6f,%.6f,%.6f,%.6e,%d,%c,%s\n",
                i, p2->x, p2->y, p2->z, p2->phase, p2->amplitude,
                p2->dimension, p2->base, geom_name);
    }

    fclose(fp);
    printf("Exported spiral to: %s\n", filename);
}

void print_statistics(const SpiralState *state) {
    printf("\n═══════════════════════════════════════════════════════════════════════════\n");
    printf("φ-SPIRAL STATISTICS\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n\n");

    // Count geometries
    int geometry_counts[8] = {0};
    double total_amplitude = 0.0;
    double min_amp = 1e100, max_amp = -1e100;

    for (size_t i = 0; i < state->count; i++) {
        int dim = state->strand1[i].dimension;
        geometry_counts[dim]++;

        double amp = fabs(state->strand1[i].amplitude);
        total_amplitude += amp;
        if (amp < min_amp) min_amp = amp;
        if (amp > max_amp) max_amp = amp;
    }

    printf("Total points: %zu (per strand)\n", state->count);
    printf("Amplitude range: %.6e to %.6e\n", min_amp, max_amp);
    printf("Mean amplitude: %.6e\n\n", total_amplitude / state->count);

    printf("Geometry Distribution:\n");
    printf("%-15s %-8s %-10s %-12s %s\n",
           "Geometry", "Dim", "Count", "Percentage", "Note");
    printf("───────────────────────────────────────────────────────────────────────────\n");

    for (int i = 0; i < 8; i++) {
        if (geometry_counts[i] > 0) {
            double pct = 100.0 * geometry_counts[i] / state->count;
            printf("%-15s %-8d %-10d %-12.2f%% %s\n",
                   GEOMETRIES[i].geometry,
                   GEOMETRIES[i].dimension,
                   geometry_counts[i],
                   pct,
                   GEOMETRIES[i].note);
        }
    }

    printf("\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char *argv[]) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("║                                                                         ║\n");
    printf("║         FASTA DNA UNIFIED FRAMEWORK V1                                 ║\n");
    printf("║         Genome-Driven φ-Spiral Evolution                               ║\n");
    printf("║                                                                         ║\n");
    printf("║  Integration: BigG + Fudge10 + Spiral8 + 8 Geometries                 ║\n");
    printf("║                                                                         ║\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("\n");

    // Initialize
    init_fibonacci_cache();

    // Load genome
    const char *genome_file = (argc > 1) ? argv[1] : "ecoli_k12.fasta";
    Genome *genome = load_genome(genome_file);
    if (!genome) {
        fprintf(stderr, "Fatal: Failed to load genome\n");
        return 1;
    }

    // Create spiral state
    size_t max_points = 8000;
    if (argc > 2) {
        max_points = atoi(argv[2]);
    }

    SpiralState *state = spiral_init(max_points);
    if (!state) {
        fprintf(stderr, "Fatal: Failed to initialize spiral state\n");
        free_genome(genome);
        return 1;
    }

    // Generate spiral
    printf("\nGenerating φ-spiral with unified D_n operator...\n");
    clock_t start = clock();

    generate_spiral(state, genome, max_points);

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Generation complete: %.3f seconds\n", elapsed);
    printf("Processing rate: %.0f points/sec\n", state->count / elapsed);

    // Statistics
    print_statistics(state);

    // Export
    const char *output_file = (argc > 3) ? argv[3] : "spiral_output.csv";
    export_to_csv(state, output_file);

    // Print D_n operator examples
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("D_n OPERATOR EXAMPLES (First 8 points)\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n\n");
    printf("%-5s %-8s %-15s %-15s %-15s\n", "Index", "Base", "Geometry", "Amplitude", "Phase");
    printf("───────────────────────────────────────────────────────────────────────────\n");

    for (size_t i = 0; i < 8 && i < state->count; i++) {
        const SpiralPoint *p = &state->strand1[i];
        printf("%-5zu %-8c %-15s %-15.6e %-15.6f\n",
               i, p->base, GEOMETRIES[p->dimension].geometry,
               p->amplitude, p->phase);
    }

    printf("\n");

    // Cleanup
    spiral_free(state);
    free_genome(genome);

    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("║                    PROCESSING COMPLETE                                  ║\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n\n");

    return 0;
}
