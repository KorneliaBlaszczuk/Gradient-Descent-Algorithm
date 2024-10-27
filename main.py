import numpy as np
from scipy.optimize import minimize
from typing import Callable
from gradient_descent import GradientDescent
from charts import charts


def f(x: np.ndarray) -> float:
    """
    f(x) = 0.5x^4 + x
    """
    return 0.5 * x[0] ** 4 + x[0]


def g(x: np.ndarray) -> float:
    """
    g(x1, x2) = 1 - 0.6 * exp(-x1^2 - (x2 + 1)^2) -
    0.4 * exp(-((x1 - 1.75)^2 + (x2 + 2)^2))
    """
    x1, x2 = x
    return (
        1
        - 0.6 * np.exp(-(x1**2) - (x2 + 1) ** 2)
        - 0.4 * np.exp(-((x1 - 1.75) ** 2 + (x2 + 2) ** 2))
    )


def get_difference_from_minimum(
    func: Callable[[np.ndarray], float], y: np.ndarray
) -> np.ndarray:
    """
    Function that returns difference between minimum
    and the value y.

    :param func: function to minimalize
    :param y: value from which we want to count difference
    :return: difference
    """
    x0 = [0, 0]
    result = minimize(func, x0)
    min_y = result.fun
    difference_y = abs(min_y - y)

    return difference_y


if __name__ == "__main__":
    initial_point_f = np.array([2.0])
    solver = GradientDescent(normalize_threshold=1e-2, learning_rate=1e-5)
    minimum_f, path_f, it_f = solver.solve(f, initial_point_f)

    print("Minimum for function f(x) for x0 = [2.0]")
    print("Found minimum of function f: ", minimum_f)
    print("Number of iterations for function f: ", it_f)
    print("Value of found minimum: ", f(minimum_f))
    print(
        "Difference in y from minimum: ", get_difference_from_minimum(f, f(minimum_f))
    )
    print(" ")

    # charts(f, path_f, minimum_f, is_2d=False)

    initial_point_f = np.array([-2.0])
    minimum_f, path_f, it_f = solver.solve(f, initial_point_f)

    print("Minimum for function f(x) for x0 = [-2.0]")
    print("Found minimum of function f: ", minimum_f)
    print("Number of iterations for function f: ", it_f)
    print("Value of found minimum: ", f(minimum_f))
    print(
        "Difference in y from minimum: ", get_difference_from_minimum(f, f(minimum_f))
    )
    print(" ")

    # charts(f, path_f, minimum_f, is_2d=False)

    initial_point_g = np.array([2.0, -3.0])
    minimum_g, path_g, it_g = solver.solve(g, initial_point_g)

    print("Minimum for function g(x) for x0 = [2.0, -3.0]")
    print("Found minimum for function g:", minimum_g)
    print("Number of iterations for function g:", it_g)
    print("Value of found minimum: ", g(minimum_g))
    print(
        "Difference in y from minimum: ", get_difference_from_minimum(g, g(minimum_g))
    )
    print(" ")

    # charts(g, path_g, minimum_g, is_2d=True)

    initial_point_g = np.array([0.0, 0.0])
    minimum_g, path_g, it_g = solver.solve(g, initial_point_g)

    print("Minimum for function g(x) for x0 = [0.0, 0.0]")
    print("Found minimum for function g:", minimum_g)
    print("Number of iterations for function g:", it_g)
    print("Value of found minimum: ", g(minimum_g))
    get_difference_from_minimum(g, g(minimum_g))
    print(
        "Difference in y from minimum: ", get_difference_from_minimum(g, g(minimum_g))
    )

    # charts(g, path_g, minimum_g, is_2d=True)
