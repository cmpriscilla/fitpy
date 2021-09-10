import logging
import math

from numbers import Number
from abc import abstractmethod
from typing import Sequence
from typing import Tuple
from typing import Optional

import numpy as np
from scipy.stats import (
    chisquare,
    uniform,
    norm,
    expon,
    gamma,
    geom,
    epps_singleton_2samp,
    binom,
    poisson,
)
from scipy.integrate import quad

from .errors import ProblematicDataError

LOGGER = logging.getLogger(__name__)


class BaseDist:
    @abstractmethod
    def fit(self, data: Sequence[Number]):
        """An abstract method to fit a distribution, given data.
        Must be defined in child classes"""
        raise NotImplementedError

    @abstractmethod
    def test_fit(self, data: Sequence[Number]):
        """An abstract method to get the goodness of fit for a distribution.
        Must be defined in child classes"""
        raise NotImplementedError

    def validate_gof(self, data: Sequence[Number], E_i: Sequence[Number]) -> None:
        """Given data and expected values for a distribution, validate whether that distribution
        can be tested via chi-squared methods.

        Args:
            data (Sequence[Number]): Input sequence, representing a particular distribution.
            E_i (Sequence[Number]): Expected bin counts for a given distribution.

        Raises:
            ProblematicDataError: If the data in question is unsuitable for a chi-squared test.
        """
        if not len(data) >= 30:
            raise ProblematicDataError("Length of dataset has to be at least 30.")
        if not (E_i >= 5).all():
            raise ProblematicDataError(
                "Number of bins (k) is too many, for this particular dataset."
            )


class Uniform(BaseDist):
    """Unif(a,b)"""

    def __init__(self, a: Optional[float] = None, b: Optional[float] = None):
        self.a = a
        self.b = b

    def fit(self, data: Sequence[Number]) -> Tuple[Number, Number]:
        """Fit a and b for a continous uniform distribution, given input data.

        Follows a MLE strategy, where a == min(data), b == max(data)

        Args:
            data (Sequence[Number]): input data, representing random variables.

        Returns:
            Tuple[Number, Number]: fitted a, b values.
        """
        self.a = min(data)
        self.b = max(data)
        return self.a, self.b

    def test_fit(
        self, data: Sequence[Number], k: Optional[int] = None
    ) -> Tuple[float, float]:
        """Given data, determine whether the fitted distribution
        has expected frequencies for a uniform distribution.

        Args:
            data (Sequence[Number]): Input data to test for fit.
            k (Optional[int]): Number of bins for chisquared test.

        Returns:
            Tuple[float, float]: test statistic and associated p-value.
        """
        if k is None:
            k = min(5, math.floor(len(data) / 5))
        E_i = np.full((1, k), len(data) / k)[0]
        intervals = [(i / k) * self.b + self.a * (1 - i / k) for i in range(k + 1)]
        O_i, _ = np.histogram(data, intervals)
        self.validate_gof(data, E_i)
        results = chisquare(O_i, E_i, 2)
        return results.statistic, results.pvalue


class Normal(BaseDist):
    """Norm(mean, variance)"""

    def __init__(self, mean: Optional[float] = None, var: Optional[float] = None):
        self.mean = mean
        self.var = var

    def fit(self, data: Sequence[Number]) -> Tuple[float, float]:
        """Fits mean and variance for a normal distribution, given input data.

        Follows a MLE strategy, where the fitted mean and variance equal the
        sample mean and sample variance, respectively.

        Args:
            data (Sequence[Number]): input data, representing random variables.

        Returns:
            Tuple[Number, Number]: fitted mean, variance.
        """
        n = len(data)
        self.mean = (1 / n) * sum(data)
        self.var = (1 / n) * sum((x - self.mean) ** 2 for x in data)
        return (self.mean, self.var)

    def test_fit(
        self, data: Sequence[Number], k: Optional[int] = None
    ) -> Tuple[float, float]:
        """Given data, determine whether the fitted distribution
        has expected frequencies for a normal distribution.

        Args:
            data (Sequence[Number]): Input data to test for fit.
            k (Optional[int]): Number of bins for chisquared test.

        Returns:
            Tuple[float, float]: test statistic and associated value.
        """
        if k is None:
            k = min(5, math.floor(len(data) / 5))
        E_i = np.full((1, k), len(data) / k)[0]
        intervals = (
            [np.NINF]
            + [
                norm.ppf(i / k, loc=self.mean, scale=np.sqrt(self.var))
                for i in range(1, k)
            ]
            + [np.inf]
        )
        O_i, _ = np.histogram(data, intervals)
        self.validate_gof(data, E_i)
        results = chisquare(O_i, E_i, 2)
        return results.statistic, results.pvalue


class Exponential(BaseDist):
    """Expon(lambda)"""

    def __init__(self, lamb: Optional[float] = None):
        self.lamb = lamb

    def validate_gt_zero(self, data: Sequence[Number]) -> None:
        """Confirm that input data is greater than 0 for an exponential distribution,
        otherwise fail.

        Args:
            data (Sequence[Number]): Input distribution

        Raises:
            ProblematicDataError: If data has some value less than or equal to 0.
        """
        if any(i <= 0 for i in data):
            raise ProblematicDataError("Exponential data cannot be less than or equal to 0.")

    def fit(self, data: Sequence[Number]) -> float:
        """Fits lambda for an exponential distribution, given input data.

        Follows a MLE strategy, where the fitted lambda value can be represented by
        the reciprocal of the sample mean.

        Args:
            data (Sequence[Number]): input data, representing random variables.

        Returns:
            float: fitted lambda value.
        """
        self.validate_gt_zero(data)
        self.lamb = len(data) / sum(data)
        return self.lamb

    def test_fit(
        self, data: Sequence[Number], k: Optional[int] = None
    ) -> Tuple[float, float]:
        """Given data, determine whether the fitted distribution
        has expected frequencies for a exponential distribution.

        Args:
            data (Sequence[Number]): Input data to test for fit.
            k (Optional[int]): Number of bins for chisquared test.

        Returns:
            Tuple[float, float]: test statistic and associated p-value.
        """
        mean = np.mean(data)
        if k is None:
            k = min(5, math.floor(len(data) / 5))
        E_i = np.full((1, k), len(data) / k)[0]
        # by invariance, mean can subsitute for 1/lambda
        intervals = [-1 * mean * math.log(1 - (i / k)) for i in range(k)] + [np.inf]
        O_i, _ = np.histogram(data, intervals)
        self.validate_gof(data, E_i)
        results = chisquare(O_i, E_i, 1)
        return results.statistic, results.pvalue


class Gamma(BaseDist):
    """Gamma(alpha, beta)"""

    def __init__(self, alpha: Optional[float] = None, beta: Optional[float] = None):
        self.alpha = alpha
        self.beta = beta

    def validate_gt_zero(self, data: Sequence[Number]) -> None:
        """Confirm that input data is greater than 0 for a gamma distribution,
        otherwise fail.

        Args:
            data (Sequence[Number]): Input distribution

        Raises:
            ProblematicDataError: If data has some values <= 0.
        """
        if any(i <= 0 for i in data):
            raise ProblematicDataError("Gamma data cannot be less than or equal to 0.")

    def trigamma(self, z: float, increment=0.00001) -> float:
        """Derive the trigamma, for some value of z.

        Args:
            z (float): Z value to get the trigamma for.
            increment (float, optional): The increment to use for deriving the trigamma. Defaults to 0.00001.

        Returns:
            float: the trigamma value at z.
        """
        digamma_result = self.digamma(z)
        return (self.digamma(z + increment) - digamma_result) / increment

    def digamma(self, z: float, increment=0.00001) -> float:
        """Derive the digamma, for some value of z.

        Args:
            z (float): Z value to get the digamma for.
            increment (float, optional): The increment to use for deriving the digamma. Defaults to 0.00001.

        Returns:
            float: the digamma value at z.
        """
        gamma_result = self.gamma_func(z)
        derivative = (self.gamma_func(z + increment) - gamma_result) / increment
        return derivative / gamma_result

    @staticmethod
    def gamma_func(z: float) -> float:
        """Calculate the gamma function for some z,
        via the integral representation.

        Args:
            z (float): Z value to get the gamma function value for.

        Returns:
            float: The value of the gamma function at z.
        """
        result, _ = quad(lambda x: np.exp(-x) * (x ** (z - 1)), 0, np.inf)
        return result

    def fit(self, data: Sequence[Number]) -> Tuple[float, float]:
        """Fits alpha and beta for an gamma distribution, given input data.

        Follows a MLE strategy, where the fitted values of alpha and beta are approximated
        via Newton's method.

        Args:
            data (Sequence[Number]): input data, representing random variables.

        Returns:
            Tuple[float, float]: fitted alpha and beta values.
        """
        self.validate_gt_zero(data)
        n = len(data)
        mean = sum(data) / n
        var = 1 / (n - 1) * sum((x - mean) ** 2 for x in data)
        logsum = sum(math.log(i) for i in data)
        threshold = 0.001
        final_value = mean / var ** (1 / 2)
        while True:
            f_of_x = (
                math.log(mean / final_value) + self.digamma(final_value) - (logsum / n)
            )
            df_of_x = -1 / final_value + self.trigamma(final_value)
            new_change = f_of_x / df_of_x
            final_value -= new_change
            if abs(f_of_x) < threshold:
                break

        self.alpha = final_value
        self.beta = mean / self.alpha
        return (self.alpha, self.beta)

    def test_fit(
        self, data: Sequence[Number], k: Optional[int] = None
    ) -> Tuple[float, float]:
        """Given data, determine whether the fitted distribution
        has expected frequencies for a gamma distribution.

        Args:
            data (Sequence[Number]): Input data to test for fit.
            k (Optional[int]): Number of bins for chisquared test.

        Returns:
            Tuple[float, float]: test statistic and associated p-value
        """
        if k is None:
            k = min(5, math.floor(len(data) / 5))
        E_i = np.full((1, k), len(data) / k)[0]
        intervals = [
            gamma.ppf(i / k, self.alpha, scale=self.beta) for i in range(k + 1)
        ]
        O_i, _ = np.histogram(data, intervals)
        self.validate_gof(data, E_i)
        results = chisquare(O_i, E_i, 2)
        return results.statistic, results.pvalue


class Geometric(BaseDist):
    """Geom(p)"""

    def __init__(self, p: Optional[float] = None):
        self.p = p

    def validate_data(self, data: Sequence[Number]) -> None:
        """Validate that input data meets conditions for a geometric distribution.

        Args:
            data (Sequence[Number]): Input random variables.

        Raises:
            ProblematicDataError: If the data is less than or equal to zero, or if the values are continous
        """
        if any(i <= 0 for i in data):
            raise ProblematicDataError(
                "Geometric data cannot be less than or equal to 0."
            )
        if any(isinstance(i, (float, np.floating)) for i in data):
            raise ProblematicDataError(
                "Geometric distributions can only take discrete data."
            )

    def fit(self, data: Sequence[Number]) -> float:
        """Fits p for an geometric distribution, given input data.

        Follows a MLE strategy, where the fitted p value can be represented by
        the reciprocal of the sample mean.

        Args:
            data (Sequence[Number]): input data, representing random variables.

        Returns:
            float: fitted p value.
        """
        self.validate_data(data)
        self.p = len(data) / sum(data)
        return self.p

    def test_fit(self, data: Sequence[Number]) -> Tuple[float, float]:
        """Given data, determine whether the fitted distribution
        has expected frequencies for a geometric distribution.

        Args:
            data (Sequence[Number]): Input data to test for fit.

        Returns:
            Tuple[float, float]: test statistic and associated p-value.
        """
        n = max(data)
        E_i = np.zeros(n)
        if n <= 2:
            raise ProblematicDataError(
                f"{n} classes is not enough to perform an appropriate chi-squared test."
            )
        for x in range(n):
            E_i[x] = len(data) * (1 - self.p) ** (x) * self.p
        xfin_edge = 0
        while E_i[xfin_edge] >= 5:
            if xfin_edge == n - 1:
                break
            xfin_edge += 1
        if xfin_edge == n - 1:
            xfin_edge = xfin_edge - 1
        E_i[xfin_edge] = sum(E_i[xfin_edge:])
        if E_i[xfin_edge] < 5:
            xfin_edge = xfin_edge - 1
            E_i[xfin_edge] = sum(E_i[xfin_edge : xfin_edge + 2])
        E_i = E_i[: (xfin_edge + 1)]
        intervals = [x + 1 for x in range(len(E_i))] + [max(data)]
        O_i, _ = np.histogram(data, intervals)
        self.validate_gof(data, E_i)
        results = chisquare(O_i, E_i, 1)
        return results.statistic, results.pvalue


class Binomial(BaseDist):
    """Binomial(n, p)"""

    def __init__(self, p: Optional[float] = None, n: Optional[int] = None):
        self.p = p
        self.n = n

    def validate_data(self, data: Sequence[Number]) -> None:
        """Validate that input data meets conditions for a binomial distribution.

        Args:
            data (Sequence[Number]): Input random variables.

        Raises:
            ProblematicDataError: If the data is less than or equal to zero, or if the values are continous
        """

        if any(i < 0 for i in data):
            raise ProblematicDataError("Binomial data cannot be less than 0.")
        if any(isinstance(i, (float, np.floating)) for i in data):
            raise ProblematicDataError(
                "Binomial distributions can only take discrete data."
            )

    def fit(self, data: Sequence[Number], n: Optional[int] = None) -> Tuple[float, int]:
        """Fits p and n for a binomial distribution, given input data.

        Follows a MLE strategy for the "known" n case specifically. If n is
        not provided, n is assumed to be the maximum value in the data set.

        p, via this formulation, is derived by mean(data)/n.

        Args:
            data (Sequence[Number]): Input data to fit
            n (Optional[int]): A estimate for n to satisfy the known n case. Defaults to None, in which case n is assumed to be == max(data)

        Returns:
            Tuple[float, int]: Returns fitted p and n values.
        """
        if n is None:
            n = max(data)
        self.n = n
        if self.n == 1:
            raise ProblematicDataError(
                "Only two classes suggests that this data is actually a single trial Bernoulli distribution, and as such will not be fit."
            )
        if not n >= max(data):
            raise ValueError("n has to be >= the largest value in the data.")
        self.p = sum(data) / (n * len(data))
        return self.p, self.n

    def test_fit(self, data: Sequence[Number]) -> Tuple[float, float]:
        """Given data, determine whether the fitted distribution
        has expected frequencies for a binomial distribution.

        If the amount of data is insufficient for a chi-squared test,
        we perform an Epps-Singleton non-parametric test.

        Args:
            data (Sequence[Number]): Input data to test for fit.

        Returns:
            Tuple[float, float]: test statistic and associated p-value.
        """
        n = self.n
        E_i = np.zeros(n + 1)
        for x in range(n + 1):
            E_i[x] = len(data) * math.comb(n, x) * (1 - self.p) ** (n - x) * self.p ** x
        xmid = math.floor(n / 2)
        xfin_edge = xmid
        while E_i[xfin_edge] >= 5:
            if xfin_edge == n:
                break
            xfin_edge += 1
        if xfin_edge == len(E_i) - 1:
            xfin_edge = xfin_edge - 1
        E_i[xfin_edge] = sum(E_i[xfin_edge:])
        if E_i[xfin_edge] < 5:
            xfin_edge = xfin_edge - 1
            E_i[xfin_edge] = sum(E_i[xfin_edge : xfin_edge + 2])
        xstart_edge = xmid - 1
        while E_i[xstart_edge] >= 5:
            if xstart_edge == 0:
                break
            xstart_edge -= 1
        if xstart_edge == 0:
            xstart_edge = xstart_edge + 1
        E_i[xstart_edge] = sum(E_i[: xstart_edge + 1])
        if E_i[xstart_edge] < 5:
            xstart_edge = xstart_edge + 1
            E_i[xstart_edge] = sum(E_i[xstart_edge - 1 : xstart_edge + 1])
        E_i = E_i[xstart_edge : xfin_edge + 1]
        intervals = [x for x in range(xstart_edge + 1, xfin_edge + 1)]
        if xstart_edge != 0:
            intervals = [0] + intervals
        if xfin_edge != n:
            intervals = intervals + [n]
        O_i, _ = np.histogram(data, intervals)
        if len(E_i) <= 2:
            LOGGER.info(
                f"Only {len(E_i)} suggests dof of <=0, falling back to a non-parametric Epps-Singleton approach"
            )
            comparison = binom.rvs(self.n, self.p, size=len(data))
            results = epps_singleton_2samp(data, comparison)
            return results.statistic, results.pvalue
        self.validate_gof(data, E_i)
        results = chisquare(O_i, E_i, 1)
        return results.statistic, results.pvalue


class Poisson(BaseDist):
    """Poisson(lambda)"""

    def __init__(self, lamb: Optional[float] = None):
        self.lamb = lamb

    def validate_data(self, data: Sequence[Number]) -> None:
        """Given input data, confirms that it meets conditions for a Poisson distribution.

        Args:
            data (Sequence[Number]): Input data to check.

        Raises:
            ProblematicDataError: If data is less than zero, non-discrete, or only 0 and 1
        """
        if any(i < 0 for i in data):
            raise ProblematicDataError("Poisson data cannot be less than 0.")
        if any(isinstance(i, (float, np.floating)) for i in data):
            raise ProblematicDataError(
                "Poisson distributions can only take discrete data."
            )
        if all(i in [0, 1] for i in np.unique(data)):
            raise ProblematicDataError(
                "This distribution only has 0 and 1 as values, which suggests a Bernoulli distribution"
            )

    def fit(self, data: Sequence[Number]) -> float:
        """Fits lambda for an poisson distribution, given input data.

        Follows a MLE strategy, where the fitted lambda value can be represented by
        the sample mean.

        Args:
            data (Sequence[Number]): input data, representing random variables.

        Returns:
            float: fitted lambda value.
        """
        self.validate_data(data)
        self.lamb = 1 / len(data) * sum(data)
        return self.lamb

    def test_fit(self, data: Sequence[Number]) -> Tuple[float, float]:
        """Given data, determine whether the fitted distribution
        has expected frequencies for a Poisson distribution.

        Args:
            data (Sequence[Number]): Input data to test for fit.

        Returns:
            Tuple[float, float]: test statistic and associated p-value.
        """
        n = max(data)
        E_i = np.zeros(n + 1)
        for x in range(n + 1):
            E_i[x] = (
                len(data)
                * math.exp(-1 * self.lamb)
                * self.lamb ** x
                / math.factorial(x)
            )
        xmid = math.floor(n / 2)
        xfin_edge = xmid
        while E_i[xfin_edge] >= 5:
            if xfin_edge == n:
                break
            xfin_edge += 1
        if xfin_edge == len(E_i) - 1:
            xfin_edge = xfin_edge - 1
        E_i[xfin_edge] = sum(E_i[xfin_edge:])
        if E_i[xfin_edge] < 5:
            xfin_edge = xfin_edge - 1
            E_i[xfin_edge] = sum(E_i[xfin_edge : xfin_edge + 2])
        xstart_edge = xmid - 1
        while E_i[xstart_edge] >= 5:
            if xstart_edge == 0:
                break
            xstart_edge -= 1
        if xstart_edge == 0:
            xstart_edge = xstart_edge + 1
        E_i[xstart_edge] = sum(E_i[: xstart_edge + 1])
        if E_i[xstart_edge] < 5:
            xstart_edge = xstart_edge + 1
            E_i[xstart_edge] = sum(E_i[xstart_edge - 1 : xstart_edge + 1])
        E_i = E_i[xstart_edge : xfin_edge + 1]
        intervals = [x for x in range(xstart_edge + 1, xfin_edge + 1)]
        if xstart_edge != 0:
            intervals = [0] + intervals
        if xfin_edge != n:
            intervals = intervals + [n]
        O_i, _ = np.histogram(data, intervals)
        self.validate_gof(data, E_i)
        results = chisquare(O_i, E_i, 1)
        return results.statistic, results.pvalue


class Bernoulli(BaseDist):
    """Bern(p)"""

    def __init__(self, p: Optional[float] = None):
        self.p = p

    def validate_data(self, data: Sequence[Number]) -> None:
        """Ensures that incoming data for the bernoulli distribution,
        is 0 or 1, as one would expect in the bernoulli case.

        Args:
            data (Sequence[Number]): Input data to check.

        Raises:
            ProblematicDataError: If data has values which are not 0 or 1
        """
        if any(i not in [0, 1] for i in data):
            raise ProblematicDataError(
                "Bernoulli distributions may only take 0 and 1 values."
            )

    def fit(self, data: Sequence[Number]) -> float:
        """Fits p for a bernoulli distribution, given input data.

        Follows a MLE strategy where p is the sample mean of a collection of 0 and 1 values.

        Args:
            data (Sequence[Number]): Input data to fit

        Returns:
            float: fitted p value.
        """
        self.validate_data(data)
        self.p = 1 / len(data) * sum(data)
        return self.p

    def test_fit(self, data: Sequence[Number]) -> Tuple[float, float]:
        """Given data, determine whether the fitted distribution
        has expected frequencies for a binomial distribution.

        We do this because any number of bernoulli distributions
        taken together is a binomial distribution. Here,
        we divide the incoming bernoulli distribution into groups of
        5, representing a Binomial(5, self.p) distribution,
        then test the fit of this generated binomial distribution.

        If the amount of data is insufficient for a chi-squared test,
        we perform an Epps-Singleton non-parametric test.

        Args:
            data (Sequence[Number]): Input data to test for fit.

        Returns:
            Tuple[float, float]: test statistic and associated p-value.
        """
        trial_size = 5
        base_binomial = [
            sum(data[i : i + trial_size]) for i in range(0, len(data), trial_size)
        ]
        binom_distribution = Binomial(p=self.p, n=trial_size)
        return binom_distribution.test_fit(base_binomial)


class Weibull(BaseDist):
    """Weibull(alpha, beta)"""

    def __init__(self, alpha: Optional[float] = None, beta: Optional[float] = None):
        self.alpha = alpha
        self.beta = beta

    def validate_gt_zero(self, data: Sequence[Number]) -> None:
        """Ensures that incoming data for the weibull distribution,
        is greater than 0

        Args:
            data (Sequence[Number]): Input data to check.

        Raises:
            ProblematicDataError: If data has values which are <=0
        """
        if any(i <= 0 for i in data):
            raise ProblematicDataError(
                "Weibull data cannot be less than or equal to 0."
            )

    @staticmethod
    def weib_anewt(data: Sequence[Number]) -> float:
        """Given a sequence of data, applies netwon approximation
        to identify the appropriate value for alpha.

        Args:
            data (Sequence[Number]): Input value to fit.

        Returns:
            float: estimated alpha value.
        """
        n = len(data)
        a = (
            (6 / math.pi ** 2)
            * (
                sum(math.log(x) ** 2 for x in data)
                - (sum(math.log(x) for x in data)) ** 2 / n
            )
            / (n - 1)
        ) ** (-1 / 2)

        def f(a, data):
            A = 1 / n * sum(math.log(x) for x in data)
            B = sum(x ** a for x in data)
            C = sum(x ** a * math.log(x) for x in data)
            return C / B - 1 / a - A

        while f(a, data) >= 0.001:
            A = 1 / n * sum(math.log(x) for x in data)
            B = sum(x ** a for x in data)
            C = sum(x ** a * math.log(x) for x in data)
            H = sum(x ** a * (math.log(x)) ** 2 for x in data)
            a = a + (A + 1 / a - C / B) / (1 / a ** 2 + (B * H - C ** 2) / B ** 2)
        return a

    def fit(self, data: Sequence[Number]) -> Tuple[float, float]:
        """Given input data, fits alpha and beta for a Weibull distribution.

        Follows a MLE strategy, where alpha and beta are estimated via the
        Newton method.

        Args:
            data (Sequence[Number]): Input data to fit.

        Returns:
            Tuple[float, float]: fitted alpha and beta values.
        """
        self.validate_gt_zero(data)
        self.alpha = self.weib_anewt(data)
        self.beta = (sum(x ** self.alpha for x in data) / len(data)) ** (1 / self.alpha)
        return (self.alpha, self.beta)

    def test_fit(
        self, data: Sequence[Number], k: Optional[int] = None
    ) -> Tuple[float, float]:
        """Given data, determine whether the fitted distribution
        has expected frequencies for a weibull distribution.

        Args:
            data (Sequence[Number]): Input data to test for fit.
            k (Optional[int]): Number of bins for chisquared test.

        Returns:
            Tuple[float, float]: test statistic and associated p-value
        """
        if k is None:
            k = min(5, math.floor(len(data) / 5))
        E_i = np.full((1, k), len(data) / k)[0]
        intervals = [
            self.beta * (-1 * math.log(1 - (i / k))) ** (1 / self.alpha)
            for i in range(k)
        ] + [np.inf]
        O_i, _ = np.histogram(data, intervals)
        self.validate_gof(data, E_i)
        results = chisquare(O_i, E_i, 2)
        return results.statistic, results.pvalue
