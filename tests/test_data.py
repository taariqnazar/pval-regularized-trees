import numpy as np

from trees.data import load_dataset, available_datasets


def test_registry_includes_new_datasets():
    names = set(available_datasets())
    assert {"boston_housing", "box_lunch", "california_housing",
            "linear_model", "neufeldt"}.issubset(names)


def test_boston_housing_shape():
    X, y = load_dataset("boston_housing")
    # paper Table 4: n=506, d=13
    assert X.shape == (506, 13)
    assert y.shape == (506,)


def test_box_lunch_shape():
    X, y = load_dataset("box_lunch")
    # paper Table 4: n=226, d=16
    assert X.shape == (226, 16)
    assert y.shape == (226,)


def test_synthetic_neufeldt():
    X, y = load_dataset("neufeldt", a=1.0, b=1.0, n=100, p=10, sigma=1.0, random_state=0)
    assert X.shape == (100, 10)
    assert y.shape == (100,)
