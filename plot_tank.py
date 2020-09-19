import numpy as np
import matplotlib.pyplot as plt
import Aircraft
from Point import Point3d


def plot_cube(center: Point3d, size: Point3d, color: str, index: int) -> None:
    x1, y1, z1 = center.x - size.x / 2, center.y - size.y / 2, center.z - size.z / 2
    x2, y2, z2 = center.x + size.x / 2, center.y + size.y / 2, center.z + size.z / 2

    ax.scatter(center.x, center.y, center.z, color=color, label=str(index))

    x = np.array([[x1, x1], [x2, x2]])
    y = np.array([[y1, y2], [y1, y2]])
    z = np.array([[z1, z1], [z1, z1]])
    ax.plot_surface(x, y, z, alpha=0.4, color=color)
    z = np.array([[z2, z2], [z2, z2]])
    ax.plot_surface(x, y, z, alpha=0.4, color=color)

    x = np.array([[x1, x1], [x1, x1]])
    y = np.array([[y1, y1], [y2, y2]])
    z = np.array([[z1, z2], [z1, z2]])
    ax.plot_surface(x, y, z, alpha=0.4, color=color)
    x = np.array([[x2, x2], [x2, x2]])
    ax.plot_surface(x, y, z, alpha=0.4, color=color)

    x = np.array([[x1, x1], [x2, x2]])
    y = np.array([[y1, y1], [y1, y1]])
    z = np.array([[z1, z2], [z1, z2]])
    ax.plot_surface(x, y, z, alpha=0.4, color=color)
    y = np.array([[y2, y2], [y2, y2]])
    ax.plot_surface(x, y, z, alpha=0.4, color=color)


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.scatter(0, 0, 0, color="black")

    colors = ["", "red", "orange", "yellow", "green", "blue", "purple"]
    for k, (tank_center, tank_size) in enumerate(zip(Aircraft.OIL_TANK_MIDDLE_POSITION, Aircraft.OIL_TANK_SIZE),
                                                 start=1):
        plot_cube(tank_center, tank_size, colors[k], k)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
