"""Question 2: demonstrate accumulated floating-point round-off error."""

import numpy as np


START_VALUE = 1.0
STEP_VALUE = 0.1
ITERATIONS = 100_000


def run_experiment(float_type):
    """Add and then subtract a typed step using the requested Python loops."""
    starting_value = float_type(START_VALUE)
    step = float_type(STEP_VALUE)
    value = starting_value

    for _ in range(ITERATIONS):
        value = float_type(value + step)
    value_after_additions = value

    for _ in range(ITERATIONS):
        value = float_type(value - step)
    final_value = value

    divergence = abs(float(final_value) - START_VALUE)
    return starting_value, step, value_after_additions, final_value, divergence


def print_experiment(float_type):
    starting_value, step, after_additions, final_value, divergence = (
        run_experiment(float_type)
    )

    print(f"\n{float_type.__name__.upper()}")
    print("-" * len(float_type.__name__))
    print(f"Starting value:                {starting_value:.17f}")
    print(f"Typed step:                    {step:.17f}")
    print(f"Number of additions:           {ITERATIONS:,}")
    print(f"Value after additions:         {after_additions:.17f}")
    print(f"Number of subtractions:        {ITERATIONS:,}")
    print(f"Final value:                   {final_value:.17f}")
    print(f"Absolute divergence from 1.0:  {divergence:.17f}")


def main():
    print("QUESTION 2 - FLOATING-POINT PRECISION")
    print("Each result uses 100,000 loop additions followed by 100,000 loop subtractions.")
    print_experiment(np.float32)
    print_experiment(np.float64)


if __name__ == "__main__":
    main()
