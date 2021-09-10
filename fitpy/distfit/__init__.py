import logging

from numbers import Number
from typing import Any
from typing import List
from typing import Sequence
from typing import Dict
from typing import Final
from typing import Type

from ..distributions import BaseDist
from ..distributions import Normal
from ..distributions import Exponential
from ..distributions import Geometric
from ..distributions import Binomial
from ..distributions import Bernoulli
from ..distributions import Poisson
from ..distributions import Gamma
from ..distributions import Uniform
from ..distributions import Weibull

from ..errors import ProblematicDataError

LOGGER = logging.getLogger(__name__)

DISTRIBUTION_MAPPING: Final[Dict[str, Type[BaseDist]]] = {
    "normal": Normal,
    "exponential": Exponential,
    "geometric": Geometric,
    "binomial": Binomial,
    "bernoulli": Bernoulli,
    "poisson": Poisson,
    "gamma": Gamma,
    "uniform": Uniform,
    "weibull": Weibull,
}

DISCRETE_DISTRIBUTIONS: Final[List[str]] = [
    "geometric",
    "binomial",
    "bernoulli",
    "poisson",
]
CONTINUOUS_DISTRIBUTIONS: Final[List[str]] = [
    "normal",
    "exponential",
    "gamma",
    "uniform",
    "weibull",
]
ALL_DISTRIBUTIONS: Final[List[str]] = DISCRETE_DISTRIBUTIONS + CONTINUOUS_DISTRIBUTIONS


def fit(data: Sequence[Number], distributions: List[str]) -> Dict[str, Any]:
    """Given a list-like object with random variables, test the fit of a list of distributions
    and return the details for the best fit distribution by p-value.

    Args:
        data (Sequence[Number]): A sequence of random variables.
        distributions (List[str]): A list of distributions which should be tested for data.

    Returns:
        Dict[str, Any]: A dictionary representing the best distribution, its test statistic, and fitted parameters.
    """
    best_result: Dict[str, Any] = {
        "test-statistic": None,
        "pvalue": 0,
        "distribution": None,
        "parameters": None,
    }
    for distribution in distributions:
        dist_fitter = DISTRIBUTION_MAPPING[distribution]
        dist_class = dist_fitter()
        try:
            dist_class.fit(data)
            chi_squared = dist_class.test_fit(data)
        except ProblematicDataError as e:
            LOGGER.warning(
                f"The {distribution} distribution was unable to fit to data with error: '{e}'"
            )
            continue
        parameters = dist_class.__dict__
        if chi_squared[1] > best_result["pvalue"]:
            best_result["test-statistic"] = chi_squared[0]
            best_result["pvalue"] = chi_squared[1]
            best_result["distribution"] = distribution
            best_result["parameters"] = parameters
    return best_result
