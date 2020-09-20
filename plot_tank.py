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


def plot_line(p1: Point3d, p2: Point3d) -> None:
    x1, y1, z1 = p1.x, p1.y, p1.z
    x2, y2, z2 = p2.x, p2.y, p2.z
    ax.plot([x1, x2], [y1, y2], [z1, z2], color="black")


def plot_2d() -> None:
    ax.scatter(0, 0, color="black")

    colors = ["", "red", "orange", "yellow", "green", "blue", "purple"]
    for k, (center, size) in enumerate(zip(Aircraft.OIL_TANK_MIDDLE_POSITION, Aircraft.OIL_TANK_SIZE),
                                       start=1):
        x1, y1 = center.x - size.x / 2, center.y - size.y / 2
        x2, y2 = center.x + size.x / 2, center.y + size.y / 2
        ax.scatter(center.x, center.y, center.z, color=colors[k], label=str(k))
        ax.fill_between([x1, x2], y1, y2, alpha=0.4, color=colors[k])
        ax.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color=colors[k])
        if k in [2, 3, 4, 5]:
            ax.plot([0, center.x], [0, center.y], color="black")

    ax.plot([Aircraft.OIL_TANK_MIDDLE_POSITION[0].x, Aircraft.OIL_TANK_MIDDLE_POSITION[1].x],
            [Aircraft.OIL_TANK_MIDDLE_POSITION[0].y, Aircraft.OIL_TANK_MIDDLE_POSITION[1].y], color="black")
    ax.plot([Aircraft.OIL_TANK_MIDDLE_POSITION[4].x, Aircraft.OIL_TANK_MIDDLE_POSITION[5].x],
            [Aircraft.OIL_TANK_MIDDLE_POSITION[4].y, Aircraft.OIL_TANK_MIDDLE_POSITION[5].y], color="black")

    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('Y', fontsize=16)
    ax.set_xlim([-8, 10])
    ax.set_xticks([-8, - 6, -4, -2, 0, 2, 4, 6, 8, 10])
    ax.set_xticklabels([-8, -6, -4, -2, 0, 2, 4, 6, 8, 10], fontsize=16)
    ax.set_ylim([-4, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_yticklabels([-4, -2, 0, 2, 4], fontsize=16)


def plot_3d() -> None:
    ax.scatter(0, 0, 0, color="black")

    colors = ["", "red", "orange", "yellow", "green", "blue", "purple"]
    for k, (tank_center, tank_size) in enumerate(zip(Aircraft.OIL_TANK_MIDDLE_POSITION, Aircraft.OIL_TANK_SIZE),
                                                 start=1):
        plot_cube(tank_center, tank_size, colors[k], k)

    plot_line(Point3d(0.0, 0.0, 0.0), Aircraft.OIL_TANK_MIDDLE_POSITION[1])
    plot_line(Point3d(0.0, 0.0, 0.0), Aircraft.OIL_TANK_MIDDLE_POSITION[2])
    plot_line(Point3d(0.0, 0.0, 0.0), Aircraft.OIL_TANK_MIDDLE_POSITION[3])
    plot_line(Point3d(0.0, 0.0, 0.0), Aircraft.OIL_TANK_MIDDLE_POSITION[4])

    plot_line(Aircraft.OIL_TANK_MIDDLE_POSITION[0], Aircraft.OIL_TANK_MIDDLE_POSITION[1])
    plot_line(Aircraft.OIL_TANK_MIDDLE_POSITION[5], Aircraft.OIL_TANK_MIDDLE_POSITION[4])

    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('Y', fontsize=16)
    ax.set_zlabel('Z', fontsize=16)
    ax.set_xlim([-6, 10])
    ax.set_xticks([-6, -4, -2, 0, 2, 4, 6, 8, 10])
    ax.set_xticklabels([-6, -4, -2, 0, 2, 4, 6, 8, 10], fontsize=16)
    ax.set_ylim([-6, 10])
    ax.set_yticks([-6, -4, -2, 0, 2, 4, 6, 8, 10])
    ax.set_yticklabels([-6, -4, -2, 0, 2, 4, 6, 8, 10], fontsize=16)
    ax.set_zlim([-6, 10])
    ax.set_zticks([-6, -4, -2, 0, 2, 4, 6, 8, 10])
    ax.set_zticklabels([-6, -4, -2, 0, 2, 4, 6, 8, 10], fontsize=16)


if __name__ == '__main__':
    fig = plt.figure()

    # ax = fig.gca(projection='3d')
    # plot_3d()

    ax = fig.gca()
    plot_2d()

    plt.show()
