#!/usr/bin/env python

"""Tests for `fitpy` package."""

import pytest

from scipy.stats import (
    expon,
    weibull_min,
    uniform,
    poisson,
    norm,
    geom,
    binom,
    bernoulli,
    gamma,
)
from fitpy import fit, DISCRETE_DISTRIBUTIONS, CONTINUOUS_DISTRIBUTIONS


def test_fit_expon():
    r = expon.rvs(size=1000, random_state=500)
    results = fit(r, distributions=CONTINUOUS_DISTRIBUTIONS)
    assert results["distribution"] == "exponential"


def test_fit_weibull():
    r = weibull_min.rvs(3, scale=1 / 2, size=1000, random_state=1000)
    results = fit(r, distributions=CONTINUOUS_DISTRIBUTIONS)
    assert results["distribution"] == "weibull"


def test_fit_gamma():
    r = gamma.rvs(3, scale=1 / 2, size=1000, random_state=1000)
    results = fit(r, distributions=CONTINUOUS_DISTRIBUTIONS)
    assert results["distribution"] == "gamma"


def test_fit_normal():
    r = norm.rvs(4, scale=1, size=1000, random_state=1000)
    results = fit(r, distributions=CONTINUOUS_DISTRIBUTIONS)
    assert results["distribution"] == "normal"


def test_fit_uniform():
    r = uniform.rvs(size=1000, random_state=1000)
    results = fit(r, distributions=CONTINUOUS_DISTRIBUTIONS)
    assert results["distribution"] == "uniform"


def test_fit_poisson():
    r = poisson.rvs(2, size=1000, random_state=1000)
    results = fit(r, distributions=DISCRETE_DISTRIBUTIONS)
    assert results["distribution"] == "poisson"


def test_fit_bernoulli():
    r = bernoulli.rvs(0.4, size=1000, random_state=1000)
    results = fit(r, distributions=DISCRETE_DISTRIBUTIONS)
    assert results["distribution"] == "bernoulli"


def test_fit_geometric():
    r = geom.rvs(0.3, size=1000, random_state=1000)
    results = fit(r, distributions=DISCRETE_DISTRIBUTIONS)
    assert results["distribution"] == "geometric"


def test_fit_binomial():
    r = binom.rvs(3, 0.4, size=1000, random_state=1000)
    results = fit(r, distributions=DISCRETE_DISTRIBUTIONS)
    assert results["distribution"] == "binomial"
