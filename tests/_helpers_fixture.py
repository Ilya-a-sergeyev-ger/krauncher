"""Fixture module for testing multi-function serialization.

Functions here simulate a user's module with helpers + task entry point.
"""


def normalize(data):
    """Helper: normalize values to [0, 1]."""
    lo, hi = min(data), max(data)
    rng = hi - lo or 1
    return [(x - lo) / rng for x in data]


def augment(batch):
    """Helper: simple data augmentation."""
    return batch + batch[::-1]


def preprocess(data):
    """Helper that calls another helper (transitive dependency)."""
    normed = normalize(data)
    return augment(normed)


def train(data, epochs=5):
    """Entry point that uses helpers from same module."""
    processed = preprocess(data)
    return {"result": sum(processed), "epochs": epochs}


def standalone(x):
    """Entry point with no helpers."""
    return x * 2
