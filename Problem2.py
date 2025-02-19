from typing import List

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import Aircraft
import BaryCenter
from Point import Point3d, distance3d


# max_distance = 0.06760845436211564


def plot_oil() -> None:
    data = pd.read_excel("Results.xlsx", "第二问结果").values
    x, y = [[], [], [], [], [], [], []], [[], [], [], [], [], [], []]
    scale = 1.7
    for time, s1, s2, s3, s4, s5, s6 in data:
        time = int(time)
        if (len(y[1]) > 0 and y[1][-1] == 1) and s1 > 0:
            x[1].append(time)
            y[1].append(1)
        if (len(y[2]) > 0 and y[2][-1] == 2) and s2 > 0:
            x[2].append(time)
            y[2].append(2)
        if (len(y[3]) > 0 and y[3][-1] == 3) and s3 > 0:
            x[3].append(time)
            y[3].append(3)
        if (len(y[4]) > 0 and y[4][-1] == 4) and s4 > 0:
            x[4].append(time)
            y[4].append(4)
        if (len(y[5]) > 0 and y[5][-1] == 5) and s5 > 0:
            x[5].append(time)
            y[5].append(5)
        if (len(y[6]) > 0 and y[6][-1] == 6) and s6 > 0:
            x[6].append(time)
            y[6].append(6)
        if (len(y[1]) > 0 and y[1][-1] > 1) or s1 > 0:
            x[1].append(time)
            y[1].append(1 + s1 / scale)
        if (len(y[2]) > 0 and y[2][-1] > 2) or s2 > 0:
            x[2].append(time)
            y[2].append(2 + s2 / scale)
        if (len(y[3]) > 0 and y[3][-1] > 3) or s3 > 0:
            x[3].append(time)
            y[3].append(3 + s3 / scale)
        if (len(y[4]) > 0 and y[4][-1] > 4) or s4 > 0:
            x[4].append(time)
            y[4].append(4 + s4 / scale)
        if (len(y[5]) > 0 and y[5][-1] > 5) or s5 > 0:
            x[5].append(time)
            y[5].append(5 + s5 / scale)
        if (len(y[6]) > 0 and y[6][-1] > 6) or s6 > 0:
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


def plot_oil_main_tank() -> None:
    data = pd.read_excel("Results.xlsx", "第二问结果").values
    x, y = [], []
    for time, s1, s2, s3, s4, s5, s6 in data:
        time = int(time)
        x.append(time)
        s = s2 + s3 + s4 + s5
        y.append(s)

    plt.fill_between(x, 0, y, color="red", alpha=0.8)

    plt.xlabel("Time (s)", fontsize=16)
    plt.xlim(0, 7200)
    plt.xticks([0, 1800, 3600, 5400, 7200], fontsize=16)
    plt.ylabel("Speed (kg/s)", fontsize=16)
    plt.yticks([0, 0.5, 1, 1.5, 2.0], [0, 0.5, 1, 1.5, 2.0], fontsize=16)

    plt.grid(axis="both")
    plt.show()


def plot_track(ideal_barycenters_np: np.array, real_barycenters_np: np.array = None) -> None:
    def plot_one_track(data: np.array, color: str, ideal_or_real: str) -> None:
        x, y, z = data[:, 0].flatten(), data[:, 1].flatten(), data[:, 2].flatten()
        ax.scatter(x, y, z, color=color, label=ideal_or_real)

        pz0 = -0.2
        xx, yy, zz = x[::100], y[::100], z[::100]
        for px, py, pz in zip(xx, yy, zz):
            ax.plot([px, px], [py, py], [pz0, pz], color="green", alpha=0.4)

        colors = ["red", "orange", "yellow", "green", "blue", "purple"]
        for k, time in enumerate([1, 1800, 3600, 5400, 7200]):
            if time < x.shape[0]:
                if ideal_or_real == "Real":
                    ax.scatter(x[time - 1], y[time - 1], z[time - 1], s=120, color=colors[k], label=f"{time}")
                else:
                    ax.scatter(x[time - 1], y[time - 1], z[time - 1], s=120, color=colors[k])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    plot_one_track(ideal_barycenters_np, "deepskyblue", "Ideal")
    if real_barycenters_np is not None:
        plot_one_track(real_barycenters_np, "yellow", "Real")

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


def evaluate_oil_plan(time: int, ideal_oil_consume_mass: float, rest_oil_mass_np: np.array) -> np.array:
    def is_meet_oil_need() -> bool:
        return np.sum(real_oil_consume_mass_np) <= -ideal_oil_consume_mass

    def oil_still_need_mass() -> float:
        return ideal_oil_consume_mass + np.sum(real_oil_consume_mass_np)

    def oil_output_from(oil_tank: int, oil_mass: float) -> bool:
        oil_mass = min(oil_mass, Aircraft.OIL_TANK_MAX_SPEED_KGps[oil_tank - 1])
        if oil_tank == 1:
            oil_mass = min(oil_mass, Aircraft.OIL_TANK_MAX_MASS[2 - 1] - rest_oil_mass_np[2 - 1])
        if oil_tank == 6:
            oil_mass = min(oil_mass, Aircraft.OIL_TANK_MAX_MASS[5 - 1] - rest_oil_mass_np[5 - 1])
        if oil_mass <= rest_oil_mass_np[oil_tank - 1]:
            real_oil_consume_mass_np[oil_tank - 1] -= oil_mass
            if oil_tank == 1:
                real_oil_consume_mass_np[2 - 1] += oil_mass
            if oil_tank == 6:
                real_oil_consume_mass_np[5 - 1] += oil_mass
            return True
        return False

    real_oil_consume_mass_np = np.zeros(6)

    # TODO: edit oil plan
    if time <= 600:
        oil_output_from(2, ideal_oil_consume_mass)
        oil_output_from(1, ideal_oil_consume_mass * 0.5)
        if not is_meet_oil_need():
            oil_output_from(4, oil_still_need_mass())
    elif time <= 1800:
        oil_output_from(2, ideal_oil_consume_mass)
        oil_output_from(1, ideal_oil_consume_mass * 0.25)
        if not is_meet_oil_need():
            oil_output_from(4, oil_still_need_mass())
    elif time <= 3000:
        oil_output_from(4, ideal_oil_consume_mass)
        if not is_meet_oil_need():
            oil_output_from(2, oil_still_need_mass())
    elif time <= 4500:
        oil_output_from(3, ideal_oil_consume_mass)
        if not is_meet_oil_need():
            oil_output_from(5, oil_still_need_mass())
    elif time <= 4900:
        oil_output_from(5, oil_still_need_mass())
        oil_output_from(6, 1.1)
        if not is_meet_oil_need() and time <= 4740:
            oil_output_from(3, oil_still_need_mass())
        if not is_meet_oil_need() and time > 4740:
            oil_output_from(4, oil_still_need_mass())
    elif time <= 5400:
        oil_output_from(5, oil_still_need_mass())
        oil_output_from(6, 0.2)
        if not is_meet_oil_need():
            oil_output_from(4, oil_still_need_mass() * (1 + 1e-8))
    elif time <= 7200:
        oil_output_from(5, oil_still_need_mass())
        if not is_meet_oil_need():
            oil_output_from(4, oil_still_need_mass() + 1e-10)

    if not is_meet_oil_need():
        raise ValueError("Real Oil < Ideal Oil")

    return real_oil_consume_mass_np


def calc(ideal_barycenters_np: np.array, ideal_oil_consume_mass_np: np.array) -> np.array:
    real_barycenters_list = []
    rest_oil_mass_np = np.array(Aircraft.OIL_TANK_INIT_OIL_MASS)
    max_distance = 0.0
    angle = 0.0
    for time in range(7200):
        time = int(time)

        # TODO: time limit
        if time > 7200:
            break

        ideal_barycenter = ideal_barycenters_np[time]
        ideal_barycenter = Point3d(ideal_barycenter[0], ideal_barycenter[1], ideal_barycenter[2])
        ideal_oil_consume_mass = ideal_oil_consume_mass_np[time]

        oil_consume_mass_np = evaluate_oil_plan(time, ideal_oil_consume_mass, rest_oil_mass_np)

        rest_oil_mass_np = rest_oil_mass_np + oil_consume_mass_np
        rest_oil_volume_np = rest_oil_mass_np / Aircraft.OIL_DENSITY_KGpm3
        description = f"time = {time:d}s"
        barycenter_oil, mass_oil = BaryCenter.calc_3d_barycenter_all_tanks(
            Aircraft.OIL_TANK_MIDDLE_POSITION, Aircraft.OIL_TANK_SIZE,
            rest_oil_volume_np.tolist(), math.radians(angle), description)
        real_barycenter, mass = BaryCenter.BaryCenter.compose3d([barycenter_oil, Point3d(0.0, 0.0, 0.0)],
                                                                [mass_oil, Aircraft.AIRCRAFT_NET_WEIGHT])
        real_barycenters_list.append([real_barycenter.x, real_barycenter.y, real_barycenter.z])
        # print(f"{time:d}\t{real_barycenter.x}\t{real_barycenter.y}\t{real_barycenter.z}")
        print(f"{time:d}\t{oil_consume_mass_np.tolist()}")

        distance = distance3d(ideal_barycenter, real_barycenter)
        max_distance = max(max_distance, distance)

    print(f"max_distance = {max_distance}")

    real_barycenters_np = np.array(real_barycenters_list)
    return real_barycenters_np


if __name__ == '__main__':
    # data_ = pd.read_excel("data.xlsx", "Problem2").values
    # np.set_printoptions(precision=3)
    # ideal_barycenters_np_ = data_[:, 1:4]
    # ideal_oil_consume_mass_np_ = data_[:, 4]
    # # plot_track(ideal_barycenters_np_, None)
    # real_barycenters_np_ = calc(ideal_barycenters_np_, ideal_oil_consume_mass_np_)
    # plot_track(ideal_barycenters_np_[:real_barycenters_np_.shape[0]], real_barycenters_np_)
    plot_oil()
    # plot_oil_main_tank()
