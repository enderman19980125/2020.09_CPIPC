import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import Aircraft
import BaryCenter
from Point import Point3d


def plot_oil(data: np.array) -> None:
    x, y = [[], [], [], [], [], [], []], [[], [], [], [], [], [], []]
    scale = 1.5
    for time, s1, s2, s3, s4, s5, s6, angle in data:
        time = int(time)
        if s1 > 0:
            x[1].append(time)
            y[1].append(1 + s1 / scale)
        if s2 > 0:
            x[2].append(time)
            y[2].append(2 + s2 / scale)
        if s3 > 0:
            x[3].append(time)
            y[3].append(3 + s3 / scale)
        if s4 > 0:
            x[4].append(time)
            y[4].append(4 + s4 / scale)
        if s5 > 0:
            x[5].append(time)
            y[5].append(5 + s5 / scale)
        if s6 > 0:
            x[6].append(time)
            y[6].append(6 + s6 / scale)

    colors = ["", "red", "orange", "yellow", "green", "blue", "purple"]
    for k in range(1, 7):
        xk, yk = x[k], y[k]
        plt.fill_between(xk, k, yk, color=colors[k])

    plt.xlabel("Time (s)", fontsize=16)
    plt.xlim(0, 7200)
    plt.xticks([0, 1800, 3600, 5400, 7200], fontsize=16)
    plt.yticks([1, 2, 3, 4, 5, 6], ["No.1 Tank", "No.2 Tank", "No.3 Tank", "No.4 Tank", "No.5 Tank", "No.6 Tank"],
               fontsize=16)

    plt.grid(axis="both")
    plt.show()


def plot_track() -> None:
    data = pd.read_excel("Results.xlsx", "第一问结果").values
    data = data[:, :4]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x, y, z = data[:, 1].flatten(), data[:, 2].flatten(), data[:, 3].flatten()
    ax.scatter(x, y, z)

    pz0 = -0.2
    xx, yy, zz = x[::100], y[::100], z[::100]
    for px, py, pz in zip(xx, yy, zz):
        ax.plot([px, px], [py, py], [pz0, pz], color="green", alpha=0.4)

    colors = ["red", "orange", "yellow", "green", "blue", "purple"]
    for k, time in enumerate([1, 1800, 3600, 5400, 7200]):
        ax.scatter(x[time - 1], y[time - 1], z[time - 1], s=120, color=colors[k], label=f"{time}s")

    # ax.set_xlabel('X', fontsize=16)
    # ax.set_ylabel('Y', fontsize=16)
    # ax.set_zlabel('Z', fontsize=16)
    ax.set_xlim([-1.2, 0.0])
    ax.set_xticks([-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0])
    ax.set_xticklabels([-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0], fontsize=16)
    ax.set_ylim([-0.6, 0.6])
    ax.set_yticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    ax.set_yticklabels([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6], fontsize=16)
    ax.set_zlim([-0.2, 1.0])
    ax.set_zticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_zticklabels([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)
    ax.legend(loc="upper left", fontsize=16)

    plt.show()


def calc(data: np.array) -> None:
    rest_oil_mass_np = np.array(Aircraft.OIL_TANK_INIT_OIL_MASS)
    for time, s1, s2, s3, s4, s5, s6, angle in data:
        time = int(time)
        oil_consume_mass_np = np.array([-s1, s1 - s2, -s3, -s4, s6 - s5, -s6])
        rest_oil_mass_np = rest_oil_mass_np + oil_consume_mass_np
        rest_oil_volume_np = rest_oil_mass_np / Aircraft.OIL_DENSITY_KGpm3
        description = f"time = {time:d}s"
        barycenter_oil, mass_oil = BaryCenter.calc_3d_barycenter_all_tanks(
            Aircraft.OIL_TANK_MIDDLE_POSITION, Aircraft.OIL_TANK_SIZE,
            rest_oil_volume_np.tolist(), math.radians(angle), description)
        barycenter, mass = BaryCenter.BaryCenter.compose3d([barycenter_oil, Point3d(0.0, 0.0, 0.0)],
                                                           [mass_oil, Aircraft.AIRCRAFT_NET_WEIGHT])
        # print(f"{time:d}\t{barycenter.x}\t{barycenter.y}\t{barycenter.z}")
        print(f"{time:d}\t{rest_oil_mass_np.tolist()}")


if __name__ == '__main__':
    # data_ = pd.read_excel("data.xlsx", sheet_name=0).values
    # plot_oil(data_)
    # calc(data_)
    plot_track()
