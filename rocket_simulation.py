
import math

import matplotlib.pyplot as plt



def velocity(vx, vy, vz):
    return math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)


def simulate(x0, y0, z0, vx0, vy0, vz0):
    k = 10
    m = 1000
    g = 10

    xarr = []
    yarr = []
    zarr = []

    x = x0
    y = y0
    z = z0

    vx = vx0
    vy = vy0
    vz = vz0

    dt = 0.5
    t = 0
    while (t < 100):
        xarr.append(x)
        yarr.append(y)
        zarr.append(z)

        v = velocity(vx, vy, vz)
        dvx = -k * v / m * vx * dt
        dvy = -k * v / m * vy * dt
        dvz = -(k * v / m * vz - g) * dt

        x += vx * dt + 1/2 * dvx * dt
        y += vy * dt + 1/2 * dvy * dt
        z += vz * dt + 1/2 * dvz * dt

        vx += dvx
        vy += dvy
        vz += dvz
        t += dt

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    print(xarr)
    # Scatter plot
    ax.scatter(xarr, yarr, zarr)

    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    fig.show()


if __name__ == '__main__':
    simulate(100, 100, 100, 10, 10, 10)


