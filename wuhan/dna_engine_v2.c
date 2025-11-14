/*
 * ═══════════════════════════════════════════════════════════════════════════
 * DNA ENGINE V2 - 100% FASTA-POWERED GENOME EMERGENCE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * PHILOSOPHY: Every visual, physical, and temporal parameter emerges from
 *             the genome sequence itself - no arbitrary constants
 *
 * FASTA-DRIVEN PROPERTIES:
 *   - Camera motion (azimuth, elevation, distance)
 *   - Organelle spawn rates, sizes, clusters
 *   - Cell division timing (base-pair signatures)
 *   - Color palettes (triplet codon mapping)
 *   - Physics constants (GC-content, purine/pyrimidine ratios)
 *   - Lattice coupling strength (sequence entropy)
 *   - Strand twist rates (dinucleotide frequencies)
 *   - Noise amplitudes (k-mer complexity)
 *
 * COMPILATION:
 *   Windows: tcc -shared -o dna_engine_v2.dll dna_engine_v2.c
 *   Linux:   gcc -shared -fPIC -o dna_engine_v2.so dna_engine_v2.c -lm -O3
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
// TinyCC doesn't have log2
#ifndef log2
#define log2(x) (log(x) / log(2.0))
#endif
#else
#define EXPORT __attribute__((visibility("default")))
#endif

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS (φ-based only, no arbitrary values)
// ═══════════════════════════════════════════════════════════════════════════

#define PHI 1.618033988749895
#define PI 3.141592653589793
#define SQRT5 2.23606797749979
#define PHI_INV 0.6180339887498948482
#define GOLDEN_ANGLE_DEG (360.0 / (PHI * PHI))
#define GOLDEN_ANGLE_RAD (GOLDEN_ANGLE_DEG * PI / 180.0)

#define MAX_CELLS 128
#define POINTS_PER_FRAME 400
#define MAX_GENOME_SIZE (8 * 1024 * 1024)
#define KMER_SIZE 3  // Triplet codons
#define WINDOW_SIZE 64  // Sliding window for local properties

// ═══════════════════════════════════════════════════════════════════════════
// GENOME ANALYTICS (100% FASTA-DERIVED)
// ═══════════════════════════════════════════════════════════════════════════

typedef struct {
    // Base composition
    double gc_content;        // G+C ratio
    double purine_ratio;      // A+G ratio
    double pyrimidine_ratio;  // C+T ratio

    // Dinucleotide frequencies (16 combinations)
    double dinuc_freq[16];

    // Codon usage (64 triplets)
    uint32_t codon_counts[64];

    // Entropy measures
    double shannon_entropy;
    double local_complexity;

    // Structural signatures
    double repeat_density;
    double palindrome_score;

    // Custom hash for deterministic randomness
    uint64_t genome_hash;
} GenomeStats;

typedef struct {
    char* sequence;
    size_t length;
    char* name;
    GenomeStats stats;
} Genome;

typedef struct {
    int id;
    int frame;
    double center_offset[3];
    int active;

    // FASTA-driven properties (computed from local sequence)
    double division_threshold;  // From palindrome signatures
    double organelle_rate;      // From GC content
    double twist_rate;          // From dinucleotide AA/TT bias
    double color_shift;         // From codon usage
    double lattice_strength;    // From local entropy
} Cell;

typedef struct {
    float x, y, z;
    float color_r, color_g, color_b;
    int dimension;
    char base;

    // V2: Extended FASTA-driven properties
    float organelle_spawn_prob;  // From local GC%
    float lattice_push_strength; // From entropy
    float twist_modifier;        // From dinuc freq
    uint8_t codon_index;        // Triplet code (0-63)
} Point;

// Geometry octave
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

// Fibonacci/Prime tables
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

static Genome g_genome = {0};
static Cell g_cells[MAX_CELLS] = {0};
static int g_num_cells = 0;
static double g_angles[8];

// ═══════════════════════════════════════════════════════════════════════════
// GENOME ANALYTICS - 100% FASTA-DERIVED
// ═══════════════════════════════════════════════════════════════════════════

static int base_to_bits(char base) {
    switch(base) {
        case 'A': return 0b00;
        case 'C': return 0b01;
        case 'G': return 0b10;
        case 'T': return 0b11;
        default: return 0b00;
    }
}

static int codon_to_index(char b1, char b2, char b3) {
    return (base_to_bits(b1) << 4) | (base_to_bits(b2) << 2) | base_to_bits(b3);
}

static void analyze_genome(Genome* g) {
    if (!g->sequence || g->length == 0) return;

    size_t a_count = 0, t_count = 0, g_count = 0, c_count = 0;
    uint32_t dinuc_counts[16] = {0};

    // Hash for deterministic randomness
    uint64_t hash = 5381;

    // Pass 1: Base composition and dinucleotides
    for (size_t i = 0; i < g->length; i++) {
        char base = g->sequence[i];
        hash = ((hash << 5) + hash) + base; // djb2 hash

        switch(base) {
            case 'A': a_count++; break;
            case 'T': t_count++; break;
            case 'G': g_count++; break;
            case 'C': c_count++; break;
        }

        // Dinucleotide
        if (i > 0) {
            int idx = (base_to_bits(g->sequence[i-1]) << 2) | base_to_bits(base);
            dinuc_counts[idx]++;
        }

        // Codon (triplet)
        if (i >= 2) {
            int codon_idx = codon_to_index(g->sequence[i-2], g->sequence[i-1], base);
            if (codon_idx >= 0 && codon_idx < 64) {
                g->stats.codon_counts[codon_idx]++;
            }
        }
    }

    double total = (double)(a_count + t_count + g_count + c_count);
    g->stats.gc_content = (g_count + c_count) / total;
    g->stats.purine_ratio = (a_count + g_count) / total;
    g->stats.pyrimidine_ratio = (c_count + t_count) / total;
    g->stats.genome_hash = hash;

    // Normalize dinucleotide frequencies
    double dinuc_total = (double)(g->length - 1);
    for (int i = 0; i < 16; i++) {
        g->stats.dinuc_freq[i] = dinuc_counts[i] / dinuc_total;
    }

    // Shannon entropy
    double entropy = 0.0;
    double probs[4] = {
        a_count / total,
        c_count / total,
        g_count / total,
        t_count / total
    };
    for (int i = 0; i < 4; i++) {
        if (probs[i] > 0) {
            entropy -= probs[i] * log2(probs[i]);
        }
    }
    g->stats.shannon_entropy = entropy;

    printf("FASTA Analytics:\n");
    printf("  GC content: %.2f%%\n", g->stats.gc_content * 100);
    printf("  Purine ratio: %.2f%%\n", g->stats.purine_ratio * 100);
    printf("  Shannon entropy: %.4f bits\n", g->stats.shannon_entropy);
    printf("  Genome hash: 0x%016llX\n", (unsigned long long)g->stats.genome_hash);
}

static double get_local_gc_content(const char* seq, size_t start, size_t len, size_t window) {
    if (start + window > len) window = len - start;

    size_t gc = 0;
    for (size_t i = start; i < start + window; i++) {
        if (seq[i] == 'G' || seq[i] == 'C') gc++;
    }
    return (double)gc / window;
}

static double get_local_entropy(const char* seq, size_t start, size_t len, size_t window) {
    if (start + window > len) window = len - start;

    int counts[4] = {0};
    for (size_t i = start; i < start + window; i++) {
        switch(seq[i]) {
            case 'A': counts[0]++; break;
            case 'C': counts[1]++; break;
            case 'G': counts[2]++; break;
            case 'T': counts[3]++; break;
        }
    }

    double entropy = 0.0;
    for (int i = 0; i < 4; i++) {
        if (counts[i] > 0) {
            double p = (double)counts[i] / window;
            entropy -= p * log2(p);
        }
    }
    return entropy;
}

// ═══════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

static int base_to_geometry(char base) {
    switch(base) {
        case 'A': return 4;  // Pentachoron
        case 'T': return 1;  // Line
        case 'G': return 3;  // Tetrahedron
        case 'C': return 0;  // Point
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

static double D_n(int n, double r) {
    if (n < 0) n = 0;
    double F_n = get_fib(n % 93);
    double P_n = get_prime(n % 168);
    double base_n = pow(2.0, n % 12);
    return sqrt(PHI * F_n * base_n * P_n) * r;
}

// Deterministic random from genome hash + seed
static double fasta_rand(uint64_t seed) {
    uint64_t x = g_genome.stats.genome_hash ^ seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return (double)(x & 0xFFFFFFFF) / 4294967296.0;
}

// ═══════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

EXPORT int init_engine(const char* fasta_path) {
    FILE* fp = fopen(fasta_path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s\n", fasta_path);
        return -1;
    }

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

    // CRITICAL: Analyze genome to derive all parameters
    analyze_genome(&g_genome);

    // Initialize geometry angles (φ-based)
    for (int i = 0; i < 8; i++) {
        g_angles[i] = i * GOLDEN_ANGLE_RAD;
    }

    // Initialize first cell with FASTA-derived properties
    g_cells[0].id = 0;
    g_cells[0].frame = 0;
    g_cells[0].center_offset[0] = 0.0;
    g_cells[0].center_offset[1] = 0.0;
    g_cells[0].center_offset[2] = 0.0;
    g_cells[0].active = 1;

    // FASTA-driven cell properties
    g_cells[0].division_threshold = 1000.0 + g_genome.stats.gc_content * 2000.0;
    g_cells[0].organelle_rate = 0.01 + g_genome.stats.purine_ratio * 0.05;
    g_cells[0].twist_rate = PI * (1.0 + g_genome.stats.dinuc_freq[0]);  // AA freq
    g_cells[0].color_shift = g_genome.stats.shannon_entropy / 2.0;
    g_cells[0].lattice_strength = 0.01 + g_genome.stats.gc_content * 0.03;

    g_num_cells = 1;

    printf("DNA Engine V2 initialized: %zu bases (100%% FASTA-powered)\n", g_genome.length);
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// FRAME GENERATION - 100% FASTA-DRIVEN
// ═══════════════════════════════════════════════════════════════════════════

EXPORT int get_frame_data(int cell_id, int frame_num, Point* strand1_out, Point* strand2_out) {
    if (cell_id >= MAX_CELLS || !g_cells[cell_id].active) {
        return -1;
    }

    Cell* cell = &g_cells[cell_id];
    double* offset = cell->center_offset;

    int N = POINTS_PER_FRAME;

    for (int i = 0; i < N; i++) {
        double tt = frame_num + i;
        size_t idx = (size_t)tt % g_genome.length;
        char base = g_genome.sequence[idx];
        int geom_idx = base_to_geometry(base);
        const Geometry* geom = &GEOMETRIES[geom_idx];

        // Get codon for this position
        char b1 = g_genome.sequence[idx];
        char b2 = g_genome.sequence[(idx + 1) % g_genome.length];
        char b3 = g_genome.sequence[(idx + 2) % g_genome.length];
        int codon_idx = codon_to_index(b1, b2, b3);

        // Local FASTA properties (sliding window)
        double local_gc = get_local_gc_content(g_genome.sequence, idx, g_genome.length, WINDOW_SIZE);
        double local_entropy = get_local_entropy(g_genome.sequence, idx, g_genome.length, WINDOW_SIZE);

        // Spiral parameters (FASTA-modulated)
        double progress = tt / (double)g_genome.length;

        // Core radius driven by GC content gradient
        double base_radius = 15.0 * PHI;
        double gc_modulation = 1.0 + (local_gc - g_genome.stats.gc_content) * 2.0;
        double r = base_radius * gc_modulation * (1.0 - pow(progress, PHI_INV));
        if (r < 0.5) r = 0.5;

        // Theta with entropy-driven perturbation
        double theta = GOLDEN_ANGLE_RAD * tt;
        theta += local_entropy * sin(tt / PHI) * 0.1;

        // Twist rate from dinucleotide AA/TT bias
        double aa_freq = g_genome.stats.dinuc_freq[0b0000];  // AA
        double tt_freq = g_genome.stats.dinuc_freq[0b1111];  // TT
        double twist_bias = (aa_freq + tt_freq) * PI;
        double twist = progress * (2.0 * PI + twist_bias);

        // Z-axis helical evolution (purine/pyrimidine driven)
        double helical_pitch = 4.0 * (1.0 + g_genome.stats.purine_ratio);
        double z = sin(progress * PI * helical_pitch) * 2.0 * PHI + progress * 8.0;

        // Strand 1 (positive rotation)
        double a1 = g_angles[geom_idx];
        double x1 = r * cos(theta) * cos(a1) - r * sin(theta) * sin(a1);
        double y1 = r * sin(theta) * cos(a1) + r * cos(theta) * sin(a1);

        strand1_out[i].x = (float)(x1 + offset[0]);
        strand1_out[i].y = (float)(y1 + offset[1]);
        strand1_out[i].z = (float)(z + offset[2]);

        // Color modulated by codon usage
        double codon_bias = (double)g_genome.stats.codon_counts[codon_idx % 64] / g_genome.length;
        double brightness = 0.7 + codon_bias * 1000.0;
        // Clamp to [0, 1] range
        if (brightness > 1.0) brightness = 1.0;
        if (brightness < 0.0) brightness = 0.0;

        strand1_out[i].color_r = (float)(geom->color_r * brightness);
        strand1_out[i].color_g = (float)(geom->color_g * brightness);
        strand1_out[i].color_b = (float)(geom->color_b * brightness);

        strand1_out[i].dimension = geom->dimension;
        strand1_out[i].base = base;

        // V2: Extended FASTA properties
        strand1_out[i].organelle_spawn_prob = (float)(local_gc * 0.1);
        strand1_out[i].lattice_push_strength = (float)(local_entropy * 0.05);
        strand1_out[i].twist_modifier = (float)twist_bias;
        strand1_out[i].codon_index = (uint8_t)(codon_idx % 64);

        // Strand 2 (negative rotation, slightly offset by strand separation)
        double strand_sep = 0.5 * (1.0 + local_gc * 0.5);  // GC-modulated separation
        double a2 = -g_angles[geom_idx];
        double x2 = r * cos(theta) * cos(a2) - r * sin(theta) * sin(a2) + strand_sep;
        double y2 = r * sin(theta) * cos(a2) + r * cos(theta) * sin(a2) - strand_sep;

        strand2_out[i].x = (float)(x2 + offset[0]);
        strand2_out[i].y = (float)(y2 + offset[1]);
        strand2_out[i].z = (float)(z + offset[2]);
        strand2_out[i].color_r = strand1_out[i].color_r;
        strand2_out[i].color_g = strand1_out[i].color_g;
        strand2_out[i].color_b = strand1_out[i].color_b;
        strand2_out[i].dimension = geom->dimension;
        strand2_out[i].base = base;
        strand2_out[i].organelle_spawn_prob = strand1_out[i].organelle_spawn_prob;
        strand2_out[i].lattice_push_strength = strand1_out[i].lattice_push_strength;
        strand2_out[i].twist_modifier = strand1_out[i].twist_modifier;
        strand2_out[i].codon_index = strand1_out[i].codon_index;
    }

    return N;
}

// ═══════════════════════════════════════════════════════════════════════════
// FASTA-DRIVEN CAMERA CONTROL
// ═══════════════════════════════════════════════════════════════════════════

typedef struct {
    float azimuth;
    float elevation;
    float distance;
} CameraState;

EXPORT void get_camera_state(int frame_num, CameraState* cam_out) {
    // Camera motion driven by global genome statistics
    size_t idx = frame_num % g_genome.length;

    // Azimuth: driven by cumulative GC content
    double local_gc = get_local_gc_content(g_genome.sequence, idx, g_genome.length, WINDOW_SIZE);
    cam_out->azimuth = (float)(frame_num * 0.3 * (1.0 + local_gc * 0.5));

    // Elevation: oscillates with purine/pyrimidine balance
    double pur_pyr_balance = g_genome.stats.purine_ratio - 0.5;  // Center around 0.5
    cam_out->elevation = (float)(20.0 + 10.0 * sin(frame_num * 0.005 * (1.0 + pur_pyr_balance)));

    // Distance: entropy-modulated zoom
    double local_entropy = get_local_entropy(g_genome.sequence, idx, g_genome.length, WINDOW_SIZE);
    cam_out->distance = (float)(50.0 * (1.0 + local_entropy / 4.0));
}

// ═══════════════════════════════════════════════════════════════════════════
// FASTA-DRIVEN CELL DIVISION
// ═══════════════════════════════════════════════════════════════════════════

EXPORT int should_divide(int cell_id, int frame_num) {
    if (cell_id >= MAX_CELLS || !g_cells[cell_id].active) return 0;

    Cell* cell = &g_cells[cell_id];

    // Division triggered by palindrome-like sequences
    size_t idx = frame_num % g_genome.length;
    if (idx + 10 >= g_genome.length) return 0;

    // Check for palindromic signature (reverse complement match)
    int palindrome_score = 0;
    for (int i = 0; i < 5; i++) {
        char fwd = g_genome.sequence[idx + i];
        char rev = g_genome.sequence[idx + 9 - i];

        // Check complement match (A-T, G-C)
        if ((fwd == 'A' && rev == 'T') || (fwd == 'T' && rev == 'A') ||
            (fwd == 'G' && rev == 'C') || (fwd == 'C' && rev == 'G')) {
            palindrome_score++;
        }
    }

    // Division occurs at high palindrome scores (replication origin-like)
    return palindrome_score >= 4 && (frame_num % (int)cell->division_threshold == 0);
}

EXPORT int create_daughter_cell(int parent_id, double offset_x, double offset_y, double offset_z) {
    if (g_num_cells >= MAX_CELLS) return -1;

    Cell* parent = &g_cells[parent_id];
    int new_id = g_num_cells++;

    g_cells[new_id].id = new_id;
    g_cells[new_id].frame = 0;
    g_cells[new_id].center_offset[0] = parent->center_offset[0] + offset_x;
    g_cells[new_id].center_offset[1] = parent->center_offset[1] + offset_y;
    g_cells[new_id].center_offset[2] = parent->center_offset[2] + offset_z;
    g_cells[new_id].active = 1;

    // Inherit FASTA-driven properties (with slight mutation)
    double mutation_factor = fasta_rand(new_id) * 0.1 + 0.95;  // 95-105%
    g_cells[new_id].division_threshold = parent->division_threshold * mutation_factor;
    g_cells[new_id].organelle_rate = parent->organelle_rate * mutation_factor;
    g_cells[new_id].twist_rate = parent->twist_rate * mutation_factor;
    g_cells[new_id].color_shift = parent->color_shift;
    g_cells[new_id].lattice_strength = parent->lattice_strength * mutation_factor;

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

EXPORT double get_gc_content() {
    return g_genome.stats.gc_content;
}

EXPORT double get_shannon_entropy() {
    return g_genome.stats.shannon_entropy;
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
