from typing import Callable
import numpy as np
from solver import Solver


class GradientDescent(Solver):
    def __init__(
        self,
        epsilon: float = 1e-4,
        learning_rate: float = 1e-2,
        normalize_threshold: float = None,
    ):
        """
        :param epsilon: tolerance level; minimal change in value of
            the funcion, below which the algorithm ends.
        :param learning_rate: controls the size of the steps taken
            during optimization to minimaze the loss function.
        :param normalize_threshold: when gradient is more than it,
            it get normalized.
        """
        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._normalize_threshold = normalize_threshold

    def get_parameters(self):
        return {
            "epsilon": self._epsilon,
            "learning_rate": self._learning_rate,
            "normalize_threshold": self._normalize_threshold,
        }

    def normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalizes vector to fit into scale [0,1].

        :param vector: vector to normalize
        :return: normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def func_gradient_value(
        self, func: Callable[[np.ndarray], float], point: np.ndarray
    ) -> np.ndarray:
        """
        Return func's gradient value at given point using
        central finite differences.

        :param func: function, for which we calculate gradient
        :param point: point, in which we calculate gradient
        :return: gradient's vector in the given point
        """
        gradient = np.zeros_like(point)
        for i in range(point.size):
            shift = np.zeros_like(point)
            shift[i] = self._epsilon
            grad = (func(point + shift) - func(point - shift)) / (2 * self._epsilon)
            gradient[i] = grad

        if (
            self._normalize_threshold is not None
            and gradient.size != 1
            and np.linalg.norm(gradient) > self._normalize_threshold
        ):
            gradient = self.normalize(gradient)

        return gradient

    def solve(
        self,
        func: Callable[[np.ndarray], float],
        initial_point: np.ndarray,
        max_iter: int = 10000,
    ) -> tuple:
        """
        Founds minimum of the given function.

        :param func: function, for which we calculate gradient
        :param initial_point: point from which we start our searching
        :param max_iter: maximum number of iterations while founding
            the minimum of the function
        :return: minimum of the function, path of the optimalization,
            number of the iterations
        """
        x = initial_point
        points = [x]
        iteration = 0
        step = self._learning_rate

        while True:
            gradient = self.func_gradient_value(func, x)
            x_new = x - step * gradient

            # If difference between our points is small, we end our algorithm
            if np.linalg.norm(x_new - x) < self._epsilon:
                break

            x = x_new
            points.append(x)
            iteration += 1

            # If number of iterations is equal to max_iter then we end
            # our algorithm
            if iteration == max_iter:
                break

        return x, points, iteration
