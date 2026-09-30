"""Question 3: benchmark NumPy vector access at different strides."""

import time
from pathlib import Path

import matplotlib
import numpy as np


# A non-interactive backend lets the script save its graph on any machine.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VECTOR_SIZE = 2**24
STRIDES = [1, 2, 4, 8, 16, 32, 64, 128]
OUTPUT_PATH = Path(__file__).with_name("numpy_vector_cache_benchmark.png")


def benchmark_stride(vector, stride):
    """Time explicit, repeated reads through a vector at one stride."""
    total = np.float64(0.0)
    elements_visited = len(range(0, vector.size, stride))

    start_time = time.perf_counter()
    for index in range(0, vector.size, stride):
        total += vector[index]
    elapsed_seconds = time.perf_counter() - start_time

    normalized_latency = elapsed_seconds / elements_visited
    return elements_visited, elapsed_seconds, normalized_latency, total


def save_graph(results):
    strides = [result[0] for result in results]
    latencies = [result[3] for result in results]

    plt.figure(figsize=(8, 5))
    plt.plot(strides, latencies, marker="o")
    plt.xscale("log", base=2)
    plt.xticks(STRIDES, [str(stride) for stride in STRIDES])
    plt.xlabel("Memory Stride Interval S")
    plt.ylabel("Normalized Latency Penalty (Seconds per Element Access)")
    plt.title("NumPy Vector Cache Benchmark")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()


def main():
    print("QUESTION 3 - NUMPY VECTOR CACHE BENCHMARK")
    print(f"Allocating one vector with 2^24 = {VECTOR_SIZE:,} float64 elements.")
    vector = np.ones(VECTOR_SIZE, dtype=np.float64)

    results = []
    for stride in STRIDES:
        visited, elapsed, latency, checksum = benchmark_stride(vector, stride)
        results.append((stride, stride * vector.itemsize, visited, latency, elapsed, checksum))

    print("\nResults (timings are machine-dependent):")
    print(
        f"{'Stride':>8} {'Bytes':>8} {'Elements visited':>18} "
        f"{'Elapsed (s)':>14} {'Seconds/element':>18} {'Checksum':>14}"
    )
    for stride, byte_distance, visited, latency, elapsed, checksum in results:
        print(
            f"{stride:8d} {byte_distance:8d} {visited:18,d} "
            f"{elapsed:14.6f} {latency:18.10e} {checksum:14.0f}"
        )

    save_graph(results)
    print(f"\nGraph saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
