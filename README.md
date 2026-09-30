# ENGG*2910 Assignment 2 — Phase 1

This repository contains the Python programs for the three Phase 1 experiments.
This README explains both how to run each program and how to reproduce its
mathematics by hand.

Only Phase 1 is documented here.

## Contents

- [`q1_vector_similarity.py`](assignment2/q1_vector_similarity.py) — converts
  sentences into Bag-of-Words vectors and compares their angles.
- [`q2_float_precision.py`](assignment2/q2_float_precision.py) — demonstrates
  accumulated floating-point round-off error.
- [`q3_cache_benchmark.py`](assignment2/q3_cache_benchmark.py) — measures the
  normalized access time for different NumPy vector strides.
- [`run_phase1.py`](assignment2/run_phase1.py) — runs the three experiments in
  order.

## Setup and commands

The project uses `uv`. From the repository root, install or synchronize the
locked dependencies with:

```bash
uv sync
```

Run all three programs with:

```bash
uv run python assignment2/run_phase1.py
```

Run one program at a time with:

```bash
uv run python assignment2/q1_vector_similarity.py
uv run python assignment2/q2_float_precision.py
uv run python assignment2/q3_cache_benchmark.py
```

The direct dependencies used by Phase 1 are NumPy and Matplotlib.

---

## Question 1: `q1_vector_similarity.py`

### What the program does

The program uses a **Bag-of-Words** model. A sentence is represented by word
counts instead of grammar or word order. Every sentence must use the same
ordered vocabulary so that a number at a particular vector index always refers
to the same word.

> **Current-source note:** At the time this README was written, `LINE1` in the
> Python file begins with `Python Python Python`. This differs from a version
> containing only one `Python`. The calculations below intentionally describe
> the current code. If `LINE1` is changed, its histogram, vector, norm, and
> angles must be recalculated.

### Step 1: Normalize and tokenize each sentence

For each line, the code first calls:

```python
sentence.lower()
```

This makes capitalization irrelevant. For example, `Python` becomes `python`.
It then applies the regular expression:

```text
[a-z]+
```

This means "one or more lowercase letters." Words are extracted while periods
and commas are ignored. The current token lists are:

```text
line1 = [python, python, python, is, an, amazing, tool, for,
         data, analysis, and, machine, learning]

line2 = [python, is, an, awesome, tool, for, data, analysis,
         and, machine, learning, tasks]

line3 = [baking, chocolate, chip, cookies, requires, flour,
         sugar, and, butter]
```

### Step 2: Make the word-frequency histograms

Count how many times each word occurs in its sentence. Most counts are `1`.
The important exception in the current source is:

```text
line1: count(python) = 3
```

The union of all unique words is then sorted alphabetically. This produces the
common vocabulary below. Reading each count column from top to bottom gives the
corresponding vector.

| Index | Vocabulary word | line1 count | line2 count | line3 count |
|---:|---|---:|---:|---:|
| 0 | amazing | 1 | 0 | 0 |
| 1 | an | 1 | 1 | 0 |
| 2 | analysis | 1 | 1 | 0 |
| 3 | and | 1 | 1 | 1 |
| 4 | awesome | 0 | 1 | 0 |
| 5 | baking | 0 | 0 | 1 |
| 6 | butter | 0 | 0 | 1 |
| 7 | chip | 0 | 0 | 1 |
| 8 | chocolate | 0 | 0 | 1 |
| 9 | cookies | 0 | 0 | 1 |
| 10 | data | 1 | 1 | 0 |
| 11 | flour | 0 | 0 | 1 |
| 12 | for | 1 | 1 | 0 |
| 13 | is | 1 | 1 | 0 |
| 14 | learning | 1 | 1 | 0 |
| 15 | machine | 1 | 1 | 0 |
| 16 | python | 3 | 1 | 0 |
| 17 | requires | 0 | 0 | 1 |
| 18 | sugar | 0 | 0 | 1 |
| 19 | tasks | 0 | 1 | 0 |
| 20 | tool | 1 | 1 | 0 |

Therefore, the three vectors are:

```text
v1 = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 3, 0, 0, 0, 1]
v2 = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1]
v3 = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
```

### Step 3: Calculate a dot product by hand

For two vectors of length `n`, the dot product is:

```text
a · b = a0b0 + a1b1 + ... + a(n-1)b(n-1)
```

Only positions where both vectors are nonzero contribute to the result.

For `v1 · v2`, nine shared words contribute `1 × 1`, while `python`
contributes `3 × 1`:

```text
v1 · v2 = 9(1 × 1) + (3 × 1)
        = 9 + 3
        = 12
```

For each comparison with `v3`, the only shared word is `and`:

```text
v1 · v3 = 1
v2 · v3 = 1
```

### Step 4: Calculate the vector lengths

The Euclidean length, or norm, of a vector is:

```text
||a|| = sqrt(a0² + a1² + ... + a(n-1)²)
```

`v1` has ten entries equal to `1` and one entry equal to `3`:

```text
||v1|| = sqrt(10(1²) + 3²)
       = sqrt(10 + 9)
       = sqrt(19)
       ≈ 4.35889894
```

`v2` has twelve entries equal to `1`:

```text
||v2|| = sqrt(12(1²))
       = sqrt(12)
       ≈ 3.46410162
```

`v3` has nine entries equal to `1`:

```text
||v3|| = sqrt(9(1²))
       = sqrt(9)
       = 3
```

### Step 5: Calculate each angle

The cosine relationship used by the code is:

```text
cos(theta) = (a · b) / (||a|| ||b||)
```

Solve for the angle with the inverse cosine:

```text
theta = arccos((a · b) / (||a|| ||b||))
```

The calculator must be in degree mode, or the radian result must be converted
using:

```text
degrees = radians × 180 / pi
```

#### line1 compared with line2

```text
cos(theta12) = 12 / (sqrt(19) × sqrt(12))
             = 12 / sqrt(228)
             ≈ 0.79471941

theta12 = arccos(0.79471941)
        ≈ 37.37123835°
```

#### line1 compared with line3

```text
cos(theta13) = 1 / (sqrt(19) × 3)
             ≈ 0.07647191

theta13 = arccos(0.07647191)
        ≈ 85.61420047°
```

#### line2 compared with line3

```text
cos(theta23) = 1 / (sqrt(12) × 3)
             ≈ 0.09622504

theta23 = arccos(0.09622504)
        ≈ 84.47816724°
```

The smallest angle is between `line1` and `line2`, so the Bag-of-Words model
considers them the most similar. A smaller angle means the vectors point in
more nearly the same direction.

Before NumPy calls `arccos`, the code clips the cosine to the interval
`[-1, 1]`. Mathematically, a cosine cannot leave this interval, but a tiny
floating-point round-off error could otherwise produce a value such as
`1.0000000000000002`, for which `arccos` is undefined.

---

## Question 2: `q2_float_precision.py`

### What the program does

The program starts at `1.0`, adds `0.1` exactly 100,000 times, and then
subtracts the typed `0.1` exactly 100,000 times. It performs the experiment
once with `np.float32` and once with `np.float64`.

The loops matter. Replacing the additions with one multiplication would perform
far fewer floating-point operations and would not demonstrate accumulated
round-off in the same way.

### The ideal hand calculation

Let:

```text
x0 = 1.0
h  = 0.1
N  = 100,000
```

With exact real-number arithmetic, the value after all additions would be:

```text
xadd = x0 + Nh
     = 1.0 + 100,000(0.1)
     = 1.0 + 10,000.0
     = 10,001.0
```

After the same number of subtractions:

```text
xfinal = xadd - Nh
       = 10,001.0 - 10,000.0
       = 1.0
```

Therefore, the ideal absolute divergence is:

```text
error = |xfinal - x0|
      = |1.0 - 1.0|
      = 0
```

### What floating-point arithmetic actually computes

The decimal fraction `0.1` has an infinitely repeating representation in
binary, just as `1/3` repeats forever in decimal. A finite floating-point value
must store a nearby approximation.

The program prints the typed steps as:

```text
float32 step ≈ 0.10000000149011612
float64 step ≈ 0.10000000000000001
```

It is useful to write each loop iteration using `fl`, meaning "round the result
to the selected floating-point type":

```text
x(0)   = fl(1.0)
x(k+1) = fl(x(k) + h),       for k = 0, 1, ..., 99,999

y(0)   = x(100,000)
y(k+1) = fl(y(k) - h),       for k = 0, 1, ..., 99,999
```

The reported error is then:

```text
error = |y(100,000) - 1.0|
```

The important point is that the computer rounds after **every** addition and
subtraction. In general:

```text
fl(fl(x + h) + h) != fl(x + 2h)
```

This is why the loop cannot be analyzed as though only one final rounding took
place.

### Results and hand verification

One validated run produced:

| Quantity | `float32` | `float64` |
|---|---:|---:|
| Starting value | 1.00000000000000000 | 1.00000000000000000 |
| Typed step | 0.10000000149011612 | 0.10000000000000001 |
| Value after additions | 9999.55273437500000000 | 10001.00000001885200618 |
| Final value | 0.99992191791534424 | 0.99999999999984246 |
| Absolute divergence | 0.00007808208465576 | 0.00000000000015754 |

The error calculation for `float32` is:

```text
error32 = |0.99992191791534424 - 1.0|
        = 0.00007808208465576
```

The error calculation for `float64` is:

```text
error64 = |0.99999999999984246 - 1.0|
        = 0.00000000000015754
```

`float32` has about 7 reliable decimal digits, while `float64` has about 15 to
16. As the running value becomes larger, the spacing between adjacent
representable numbers also becomes larger. Repeated rounding therefore affects
`float32` much more strongly in this experiment.

Do not round the program values before calculating the divergence. Rounding
the final value to `1.0` would hide the error that the experiment is meant to
show.

---

## Question 3: `q3_cache_benchmark.py`

### What the program does

The program allocates one NumPy vector containing exactly:

```text
N = 2²⁴ = 16,777,216 elements
```

Every element has type `float64`, and a `float64` occupies 8 bytes. The vector's
data therefore occupies:

```text
16,777,216 elements × 8 bytes/element
= 134,217,728 bytes
= 128 MiB
```

The vector is filled with ones. For each stride `S`, the loop reads:

```text
V[0], V[S], V[2S], V[3S], ...
```

and adds every visited value to a checksum.

### Calculate the number of accesses

For a vector with `N` elements and stride `S`, the visited indices have the
form:

```text
index(k) = kS
```

and must satisfy:

```text
kS < N
```

The exact number of visited elements is:

```text
elements visited = floor((N - 1) / S) + 1
                 = ceiling(N / S)
```

All required strides divide `2²⁴` exactly, so this experiment can use the
simpler calculation:

```text
elements visited = N / S
```

### Calculate the byte distance

Each element occupies 8 bytes, so moving forward by `S` elements moves forward
by:

```text
byte distance = S × 8 bytes
```

The complete hand-calculated table is:

| Stride `S` | Byte distance `8S` | Elements visited `N/S` |
|---:|---:|---:|
| 1 | 8 bytes | 16,777,216 |
| 2 | 16 bytes | 8,388,608 |
| 4 | 32 bytes | 4,194,304 |
| 8 | 64 bytes | 2,097,152 |
| 16 | 128 bytes | 1,048,576 |
| 32 | 256 bytes | 524,288 |
| 64 | 512 bytes | 262,144 |
| 128 | 1,024 bytes | 131,072 |

For example, with `S = 32`:

```text
byte distance    = 32 × 8
                 = 256 bytes

elements visited = 16,777,216 / 32
                 = 524,288
```

### Verify the checksum

Every vector element is `1.0`. If `M` elements are visited, the checksum should
be:

```text
checksum = 1.0 + 1.0 + ... + 1.0
         = M
```

For stride `32`, the expected checksum is therefore `524,288`. This makes the
checksum a simple check that every intended read was used.

### Calculate normalized latency

For each stride, `time.perf_counter()` measures the total loop time:

```text
elapsed time = finish time - start time
```

The program then removes the effect of different access counts by dividing the
total time by the number of elements visited:

```text
normalized latency = elapsed seconds / elements visited
```

For a worked example, suppose stride `32` took `0.100000` seconds. This example
number is illustrative; actual timing must come from the computer running the
program.

```text
normalized latency
    = 0.100000 s / 524,288 accesses
    ≈ 1.90735 × 10⁻⁷ s/access
```

To verify any printed row by hand, copy its elapsed time and divide it by the
`elements visited` value from the table above.

### Interpret the graph

The graph plots:

```text
x = stride S, on a base-2 logarithmic scale
y = elapsed seconds / elements visited
```

Small strides access nearby memory locations, while large strides jump farther
between reads. Computer caches load memory in blocks, so nearby accesses may be
able to reuse data already fetched into a cache. Larger strides can reduce that
spatial locality. The exact curve depends on the processor, cache hierarchy,
operating-system scheduling, Python overhead, and other programs running at the
same time. The code therefore measures the machine instead of hard-coding an
expected result.

The generated graph is saved at:

```text
assignment2/numpy_vector_cache_benchmark.png
```

Regenerate it on the laptop used for the assignment or lab defense:

```bash
uv run python assignment2/q3_cache_benchmark.py
```

---

## Combined runner: `run_phase1.py`

This file does not introduce another numerical method. It imports the other
three modules and calls their `main()` functions in this order:

```text
q1_vector_similarity.main()
        ↓
q2_float_precision.main()
        ↓
q3_cache_benchmark.main()
```

Lines of `=` are printed between the outputs so that the three experiments are
easy to distinguish. Running this file is mathematically equivalent to running
the three individual commands in the same order.

## Summary of formulas

| Experiment | Formula |
|---|---|
| Bag-of-Words dot product | `a · b = sum(ai × bi)` |
| Vector norm | `||a|| = sqrt(sum(ai²))` |
| Angle between vectors | `theta = arccos((a · b) / (||a|| ||b||))` |
| Ideal repeated addition | `xadd = x0 + Nh` |
| Floating-point divergence | `error = |xfinal - x0|` |
| Vector storage | `bytes = element count × 8` |
| Elements visited | `ceiling(N / S)` |
| Byte distance | `S × 8` |
| Normalized latency | `elapsed seconds / elements visited` |
