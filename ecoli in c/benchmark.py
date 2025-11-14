#!/usr/bin/env python3
"""
Performance Benchmark: Native C vs Python (fasta4.py)
Compare execution times for φ-spiral generation
"""

import time
import subprocess
import os

def benchmark_c_version(points=8000):
    """Benchmark native C implementation"""
    print("="*70)
    print("BENCHMARKING NATIVE C VERSION")
    print("="*70)

    start = time.time()
    result = subprocess.run([
        './fasta_dna_unified_v1.exe',
        'ecoli_k12.fasta',
        str(points),
        'spiral_output_bench.csv'
    ], capture_output=True, text=True)
    elapsed = time.time() - start

    # Extract timing from output
    for line in result.stdout.split('\n'):
        if 'seconds' in line and 'Generation' in line:
            # Parse the actual processing time
            parts = line.split(':')
            if len(parts) > 1:
                gen_time = float(parts[1].strip().split()[0])
                print(f"Generation time: {gen_time:.6f} seconds")
                print(f"Total runtime: {elapsed:.6f} seconds")
                return gen_time, elapsed

    return elapsed, elapsed

def benchmark_python_version(points=8000):
    """Benchmark Python version (if available)"""
    if not os.path.exists('fasta4.py'):
        print("\nfasta4.py not found - skipping Python benchmark")
        return None, None

    print("\n" + "="*70)
    print("BENCHMARKING PYTHON VERSION")
    print("="*70)

    start = time.time()
    result = subprocess.run([
        'python', 'fasta4.py',
        'ecoli_k12.fasta',
        str(points)
    ], capture_output=True, text=True)
    elapsed = time.time() - start

    print(f"Total runtime: {elapsed:.6f} seconds")
    return elapsed, elapsed

def main():
    print("\nφ-SPIRAL PERFORMANCE BENCHMARK")
    print("E. coli K-12 Genome (4.6M bases)")
    print("Points to generate: 8000\n")

    # Run benchmarks
    c_gen_time, c_total_time = benchmark_c_version(8000)
    py_gen_time, py_total_time = benchmark_python_version(8000)

    # Comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)

    print(f"\nNative C (TinyCC):")
    print(f"  Generation: {c_gen_time:.6f} seconds")
    print(f"  Total:      {c_total_time:.6f} seconds")
    print(f"  Throughput: {8000/c_gen_time:,.0f} points/sec")

    if py_total_time:
        print(f"\nPython (fasta4.py):")
        print(f"  Total: {py_total_time:.6f} seconds")

        speedup = py_total_time / c_total_time
        print(f"\n🚀 SPEEDUP: {speedup:.1f}x faster")

        time_saved = py_total_time - c_total_time
        print(f"   Time saved: {time_saved:.3f} seconds ({time_saved/py_total_time*100:.1f}%)")

        # Extrapolate to full genome
        if 8000 < 4641652:
            full_genome_c = c_gen_time * (4641652 / 8000)
            full_genome_py = py_total_time * (4641652 / 8000)

            print(f"\n📊 FULL GENOME PROJECTION (4.6M bases):")
            print(f"   Native C:  {full_genome_c:.2f} seconds ({full_genome_c/60:.2f} minutes)")
            print(f"   Python:    {full_genome_py:.2f} seconds ({full_genome_py/60:.2f} minutes)")
            print(f"   Savings:   {(full_genome_py-full_genome_c)/60:.2f} minutes")
    else:
        print("\n(Python version not available for comparison)")

    print("\n" + "="*70)

if __name__ == '__main__':
    main()
