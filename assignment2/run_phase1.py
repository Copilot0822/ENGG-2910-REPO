"""Run all three ENGG*2910 Assignment 2 Phase 1 experiments."""

import q1_vector_similarity
import q2_float_precision
import q3_cache_benchmark


def main():
    q1_vector_similarity.main()
    print("\n" + "=" * 80 + "\n")
    q2_float_precision.main()
    print("\n" + "=" * 80 + "\n")
    q3_cache_benchmark.main()


if __name__ == "__main__":
    main()
