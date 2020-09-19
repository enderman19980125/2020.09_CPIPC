from typing import List

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import Aircraft
import BaryCenter
from Point import Point3d, distance3d


def plot_track(data: np.array) -> None:
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

    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('Y', fontsize=16)
    ax.set_zlabel('Z', fontsize=16)
    ax.set_xlim([-2.0, 0.0])
    ax.set_xticks([-2.0, -1.6, -1.2, -0.8, -0.4, 0.0])
    ax.set_xticklabels([-2.0, -1.6, -1.2, -0.8, -0.4, 0.0], fontsize=16)
    ax.set_ylim([-1.0, 1.0])
    ax.set_yticks([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
    ax.set_yticklabels([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0], fontsize=16)
    ax.set_zlim([-0.4, 1.6])
    ax.set_zticks([-0.4, 0.0, 0.4, 0.8, 1.2, 1.6])
    ax.set_zticklabels([-0.4, 0.0, 0.4, 0.8, 1.2, 1.6], fontsize=16)
    ax.legend(loc="upper left", fontsize=16)

    plt.show()


def evalute_oil_plan(ideal_barycenter: Point3d, real_barycenter: Point3d, ideal_oil_consume_mass: float) -> np.array:
    pass


def calc(ideal_barycenters_np: np.array, ideal_oil_consume_mass_np: np.array) -> List[Point3d]:
    barycenters_list = []
    rest_oil_mass_np = np.array(Aircraft.OIL_TANK_INIT_OIL_MASS)
    max_distance = 0.0
    angle = 0.0
    barycenter = Point3d(0.0, 0.0, 0.0)
    for time in range(7200):
        time = int(time)
        ideal_barycenter = ideal_barycenters_np[time]
        ideal_barycenter = Point3d(ideal_barycenter[0], ideal_barycenter[1], ideal_barycenter[2])
        ideal_oil_consume_mass = ideal_oil_consume_mass_np[time]

        oil_consume_mass_np = evalute_oil_plan(ideal_barycenter, barycenter, ideal_oil_consume_mass)

        rest_oil_mass_np = rest_oil_mass_np + oil_consume_mass_np
        rest_oil_volume_np = rest_oil_mass_np / Aircraft.OIL_DENSITY_KGpm3
        description = f"time = {time:d}s"
        barycenter_oil, mass_oil = BaryCenter.calc_3d_barycenter_all_tanks(
            Aircraft.OIL_TANK_MIDDLE_POSITION, Aircraft.OIL_TANK_SIZE,
            rest_oil_volume_np.tolist(), math.radians(angle), description)
        barycenter, mass = BaryCenter.BaryCenter.compose3d([barycenter_oil, Point3d(0.0, 0.0, 0.0)],
                                                           [mass_oil, Aircraft.AIRCRAFT_NET_WEIGHT])
        barycenters_list.append(barycenter)
        # print(f"{time:d}\t{barycenter.x}\t{barycenter.y}\t{barycenter.z}")
        print(f"{time:d}\t{rest_oil_mass_np.tolist()}")

        distance = distance3d(ideal_barycenter, barycenter)
        max_distance = max(max_distance, distance)

    print(f"max_distance = {max_distance}")

    return barycenters_list


if __name__ == '__main__':
    data_ = pd.read_excel("data.xlsx", "Problem2").values
    ideal_barycenters_np_ = data_[:, 1:4]
    ideal_oil_consume_mass_np_ = data_[:, 4]
    # plot_oil(data_)
    real_barycenters_list_ = calc(ideal_barycenters_np_, ideal_oil_consume_mass_np_)
    plot_track(data_, real_barycenters_list_)
