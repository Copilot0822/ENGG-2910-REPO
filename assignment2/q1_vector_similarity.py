"""Question 1: compare sentences using Bag-of-Words vectors."""

import re
from collections import Counter

import numpy as np


LINE1 = "Python Python Python is an amazing tool for data analysis and machine learning."
LINE2 = "Python is an awesome tool for data analysis and machine learning tasks."
LINE3 = "Baking chocolate chip cookies requires flour, sugar, and butter."


def tokenize(sentence):
    """Lowercase a sentence and extract words, leaving punctuation behind."""
    return re.findall(r"[a-z]+", sentence.lower())


def angle_between(vector_a, vector_b):
    """Return the angle and intermediate dot-product values for two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    cosine = dot_product / (norm_a * norm_b)

    # Round-off can put the result just outside the valid arccos range.
    cosine = np.clip(cosine, -1.0, 1.0)
    angle_degrees = np.degrees(np.arccos(cosine))
    return angle_degrees, dot_product, norm_a, norm_b, cosine


def main():
    print("QUESTION 1 - VECTOR SIMILARITY")
    print("Words are lowercased and extracted with the regular expression [a-z]+.")

    lines = {"line1": LINE1, "line2": LINE2, "line3": LINE3}
    tokens = {name: tokenize(line) for name, line in lines.items()}
    histograms = {name: Counter(words) for name, words in tokens.items()}

    # Alphabetical order makes the shared vector positions easy to explain.
    vocabulary = sorted(set().union(*tokens.values()))
    vectors = {
        name: np.array([histogram[word] for word in vocabulary], dtype=int)
        for name, histogram in histograms.items()
    }

    print("\nCommon vocabulary (index: word):")
    for index, word in enumerate(vocabulary):
        print(f"  {index:2}: {word}")

    print("\nVocabulary as an ordered list:")
    print(vocabulary)

    for name in lines:
        print(f"\n{name}: {lines[name]}")
        print(f"  tokens:    {tokens[name]}")
        print(f"  histogram: {dict(sorted(histograms[name].items()))}")
        print(f"  vector:    {vectors[name].tolist()}")

    comparisons = [
        ("line1", "line2"),
        ("line1", "line3"),
        ("line2", "line3"),
    ]
    angles = {}

    print("\nAngle calculations using cos(theta) = (a dot b) / (||a|| ||b||):")
    for name_a, name_b in comparisons:
        angle, dot_product, norm_a, norm_b, cosine = angle_between(
            vectors[name_a], vectors[name_b]
        )
        angles[(name_a, name_b)] = angle
        print(f"\n{name_a} compared with {name_b}:")
        print(f"  dot product = {dot_product}")
        print(f"  ||{name_a}|| = {norm_a:.8f}")
        print(f"  ||{name_b}|| = {norm_b:.8f}")
        print(f"  cosine      = {cosine:.8f}")
        print(f"  angle       = {angle:.8f} degrees")

    most_similar = min(angles, key=angles.get)
    print(
        f"\n{most_similar[0]} and {most_similar[1]} have the smallest angle "
        f"({angles[most_similar]:.8f} degrees), so they are the most similar "
        "under this Bag-of-Words representation."
    )


if __name__ == "__main__":
    main()
