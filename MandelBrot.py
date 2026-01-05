import numpy as np
import matplotlib.pyplot as plt


def mandelbrot(xmin=-2, xmax=2, ymin=-2, ymax=2, nx=1000, ny=1000,
               bound=2.0, max_iterations=1000, p=2):
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    c = X + 1j * Y

    z = np.zeros_like(c, dtype=np.complex128)
    counts = np.zeros(c.shape, dtype=np.int32)
    escaped = np.zeros(c.shape, dtype=bool)
    bound_sq = bound * bound

    for i in range(1, max_iterations + 1):
        not_escaped = ~escaped
        if not np.any(not_escaped):
            break

        # update only points that haven't escaped yet
        z[not_escaped] = z[not_escaped] ** p + c[not_escaped]

        # use squared magnitude to avoid sqrt
        mag_sq = (z.real * z.real + z.imag * z.imag)
        newly_escaped = (mag_sq >= bound_sq) & (~escaped)
        counts[newly_escaped] = i
        escaped |= newly_escaped

    return x, y, counts


if __name__ == "__main__":
    x_domain, y_domain, iteration_array = mandelbrot(nx=1000, ny=1000,
                                                     max_iterations=1000, p=2)

    ax = plt.axes()
    ax.set_aspect("equal")
    graph = ax.pcolormesh(x_domain, y_domain, iteration_array, cmap="nipy_spectral")
    plt.colorbar(graph)
    plt.xlabel("Real-Axis")
    plt.ylabel("Imaginary-Axis")
    plt.show()
