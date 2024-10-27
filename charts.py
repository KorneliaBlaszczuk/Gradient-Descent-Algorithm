import numpy as np
import matplotlib.pyplot as plt


def charts(func, points, solution, is_2d=False):
    # Set up the figure
    plt.figure(figsize=(15, 10))  # Size of the window
    points = np.array(points)

    if is_2d:
        x = np.linspace(-4, 4, 100)
        y = np.linspace(-4, 4, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[func(np.array([x_, y_])) for x_ in x] for y_ in y])

        # Plot the contour
        contour = plt.contour(X, Y, Z, levels=50, cmap="viridis")
        plt.colorbar(contour, label="Y")

        # Plot the path and solution
        plt.plot(
            points[:, 0],
            points[:, 1],
            marker="o",
            color="red",
            markersize=2,
            label="Path",
        )
        plt.scatter(solution[0], solution[1], color="blue", label="Solution", zorder=5)

        plt.xlabel("X1")
        plt.ylabel("X2")
    else:
        x = np.linspace(-3, 3, 100)
        y = np.array([func(np.array([x_])) for x_ in x])

        # Plot the function
        plt.plot(x, y, color="gray", alpha=0.5)

        # Plot the path and solution
        plt.plot(
            points[:, 0],
            [func(np.array([p])) for p in points[:, 0]],
            marker="o",
            color="red",
            markersize=4,
            label="Path",
        )
        plt.scatter(
            solution[0], func(solution), color="blue", label="Solution", zorder=5
        )

        plt.xlabel("X")
        plt.ylabel("Y")

    plt.title("Gradient Descent Optimization Path")
    plt.legend()
    plt.show()
