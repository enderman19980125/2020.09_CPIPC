import math
import matplotlib.pyplot as plt
from typing import Tuple, List
from Point import Point2d, Point3d
from Aircraft import OIL_DENSITY_KGpm3


class Geometry:
    @staticmethod
    def get_linear_equation_from_two_points(p1: Point2d, p2: Point2d) -> Tuple[float, float, float]:
        a = p2.y - p1.y
        b = p1.x - p2.x
        c = p2.x * p1.y - p1.x * p2.y
        return a, b, c

    @staticmethod
    def get_distance_between_two_points(p1: Point2d, p2: Point2d) -> float:
        distance = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
        return distance

    @staticmethod
    def get_distance_between_point_and_line(p: Point2d, line: Tuple[float, float, float]) -> float:
        a, b, c = line
        distance = math.fabs(a * p.x + b * p.y + c) / math.sqrt(a ** 2 + b ** 2)
        return distance

    @staticmethod
    def get_distance_between_point_and_pline(p: Point2d, line_p1: Point2d, line_p2: Point2d) -> float:
        line = Geometry.get_linear_equation_from_two_points(line_p1, line_p2)
        distance = Geometry.get_distance_between_point_and_line(p, line)
        return distance

    @staticmethod
    def get_point_in_pline_for_given_y(y: float, line_p1: Point2d, line_p2: Point2d) -> Point2d:
        a, b, c = Geometry.get_linear_equation_from_two_points(line_p1, line_p2)
        x = (-b * y - c) / a
        p = Point2d(x, y)
        return p

    @staticmethod
    def is_point_in_cuboid(p: Point3d, cuboid_middle: Point3d, cuboid_size: Point3d) -> bool:
        pl = cuboid_middle - cuboid_size / 2
        ph = cuboid_middle + cuboid_size / 2
        return pl.x <= p.x <= ph.x and pl.y <= p.y <= ph.y and pl.z <= p.z <= ph.z


class Area:
    @staticmethod
    def triangle(p1: Point2d, p2: Point2d, p3: Point2d) -> float:
        a = Geometry.get_distance_between_two_points(p1, p2)
        b = Geometry.get_distance_between_two_points(p1, p3)
        c = Geometry.get_distance_between_two_points(p2, p3)
        p = (a + b + c) / 2
        s2 = p * (p - a) * (p - b) * (p - c)
        s2 = -s2 if -1e-6 < s2 < 0 else s2
        s = math.sqrt(s2)
        return s

    @staticmethod
    def parallelogram(p1: Point2d, p2: Point2d, p3: Point2d, p4: Point2d) -> float:
        a1 = Geometry.get_distance_between_two_points(p1, p2)
        a2 = Geometry.get_distance_between_two_points(p3, p4)
        h1 = Geometry.get_distance_between_point_and_pline(p3, p1, p2)
        h2 = Geometry.get_distance_between_point_and_pline(p4, p1, p2)
        if math.fabs(a1 - a2) > 1e-6 or math.fabs(h1 - h2) > 1e-6:
            raise ValueError("NOT A Parallelogram")
        s = a1 * h1
        return s


class BaryCenter:
    @staticmethod
    def triangle(p1: Point2d, p2: Point2d, p3: Point2d) -> Tuple[Point2d, float]:
        x = (p1.x + p2.x + p3.x) / 3
        y = (p1.y + p2.y + p3.y) / 3
        c = Point2d(x, y)
        s = Area.triangle(p1, p2, p3)
        return c, s

    @staticmethod
    def polygon4(p1: Point2d, p2: Point2d, p3: Point2d, p4: Point2d) -> Tuple[Point2d, float]:
        c1, s1 = BaryCenter.triangle(p1, p2, p3)
        c2, s2 = BaryCenter.triangle(p1, p3, p4)
        s = s1 + s2
        c = (s1 * c1 + s2 * c2) / s
        return c, s

    @staticmethod
    def polygon5(p1: Point2d, p2: Point2d, p3: Point2d, p4: Point2d, p5: Point2d) -> Tuple[Point2d, float]:
        c1, s1 = BaryCenter.polygon4(p1, p2, p3, p4)
        c2, s2 = BaryCenter.triangle(p1, p4, p5)
        s = s1 + s2
        c = (s1 * c1 + s2 * c2) / s
        return c, s

    @staticmethod
    def polygon7(p1: Point2d, p2: Point2d, p3: Point2d, p4: Point2d, p5: Point2d, p6: Point2d,
                 p7: Point2d) -> Tuple[Point2d, float]:
        c1, s1 = BaryCenter.polygon5(p1, p2, p3, p4, p5)
        c2, s2 = BaryCenter.polygon4(p1, p5, p6, p7)
        s = s1 + s2
        c = (s1 * c1 + s2 * c2) / s
        return c, s

    @staticmethod
    def compose2d(barycenters: List[Point2d], masses: List[float]) -> Tuple[Point2d, float]:
        c = 0.0 * barycenters[0]
        s = 0.0
        for barycenter, mass in zip(barycenters, masses):
            c = c + mass * barycenter
            s += mass
        c = c / s
        return c, s

    @staticmethod
    def compose3d(barycenters: List[Point3d], masses: List[float]) -> Tuple[Point3d, float]:
        c = 0.0 * barycenters[0]
        s = 0.0
        for barycenter, mass in zip(barycenters, masses):
            c = c + mass * barycenter
            s += mass
        c = c / s
        return c, s


def _tank_status_and_control_points(p0: Point2d, p1: Point2d, p2: Point2d, p3: Point2d) \
        -> Tuple[str, tuple, tuple, tuple]:
    if 0 <= p1.y <= p3.y:
        tank_status = "U1"
        pp1 = Geometry.get_point_in_pline_for_given_y(p1.y, p0, p3)
        pp3 = Geometry.get_point_in_pline_for_given_y(p3.y, p1, p2)
        b0, b1, b2, m0, m1, m2, m3, t0, t1, t2 = p0, pp1, p1, pp1, p1, pp3, p3, p3, pp3, p2
    elif 0 <= p3.y < p1.y:
        tank_status = "U2"
        pp1 = Geometry.get_point_in_pline_for_given_y(p1.y, p2, p3)
        pp3 = Geometry.get_point_in_pline_for_given_y(p3.y, p0, p1)
        b0, b1, b2, m0, m1, m2, m3, t0, t1, t2 = p0, p3, pp3, p3, pp3, p1, pp1, pp1, p1, p2
    elif p1.y < 0 <= p2.y:
        tank_status = "D1"
        pp0 = Geometry.get_point_in_pline_for_given_y(p0.y, p1, p2)
        pp2 = Geometry.get_point_in_pline_for_given_y(p2.y, p0, p3)
        b0, b1, b2, m0, m1, m2, m3, t0, t1, t2 = p1, p0, pp0, p0, pp0, p2, pp2, pp2, p2, p3
    elif p1.y < p2.y < 0:
        tank_status = "D2"
        pp0 = Geometry.get_point_in_pline_for_given_y(p0.y, p2, p3)
        pp2 = Geometry.get_point_in_pline_for_given_y(p2.y, p0, p1)
        b0, b1, b2, m0, m1, m2, m3, t0, t1, t2 = p1, pp2, p2, pp2, p2, pp0, p0, p0, pp0, p3
    else:
        raise ValueError("Unexpected Tank Status")
    return tank_status, (b0, b1, b2), (m0, m1, m2, m3), (t0, t1, t2)


def _oil_status_and_barycenter(bottom_triangle: tuple, middle_parallelogram: tuple, top_triangle: tuple,
                               oil_area: float) -> Tuple[int, Point2d, float, Point2d, Point2d]:
    (b0, b1, b2), (m0, m1, m2, m3), (t0, t1, t2) = bottom_triangle, middle_parallelogram, top_triangle
    s1 = Area.triangle(b0, b1, b2)
    s2 = Area.parallelogram(m0, m1, m2, m3)
    if oil_area <= s1:
        oil_status = 1
        triangle_a = Geometry.get_distance_between_two_points(b1, b2)
        triangle_h = Geometry.get_distance_between_point_and_pline(b0, b1, b2)
        h = math.sqrt(2 * oil_area * triangle_h / triangle_a)
        o1 = Geometry.get_point_in_pline_for_given_y(b0.y + h, b0, b1)
        o2 = Geometry.get_point_in_pline_for_given_y(b0.y + h, b0, b2)
        c, s = BaryCenter.triangle(b0, o1, o2)
    elif oil_area <= s1 + s2:
        oil_status = 2
        parallelogram_a = Geometry.get_distance_between_two_points(m0, m1)
        h = (oil_area - s1) / parallelogram_a
        o1 = Geometry.get_point_in_pline_for_given_y(m0.y + h, m0, m3)
        o2 = Geometry.get_point_in_pline_for_given_y(m0.y + h, m1, m2)
        c, s = BaryCenter.polygon5(b0, b2, o2, o1, b1)
    else:
        oil_status = 3
        triangle_a = Geometry.get_distance_between_two_points(t0, t1)
        triangle_h = Geometry.get_distance_between_point_and_pline(t2, t0, t1)
        h = triangle_h - math.sqrt(triangle_h ** 2 - 2 * (oil_area - s1 - s2) * triangle_h / triangle_a)
        o1 = Geometry.get_point_in_pline_for_given_y(t0.y + h, t0, t2)
        o2 = Geometry.get_point_in_pline_for_given_y(t0.y + h, t1, t2)
        c, s = BaryCenter.polygon7(b0, m1, m2, o2, o1, m3, m0)
    return oil_status, c, s, o1, o2


def _plot_2d(tank_points: tuple, bottom_triangle: tuple, middle_parallelogram: tuple, top_triangle: tuple,
             barycenter: Point2d, o1: Point2d, o2: Point2d,
             length: float, height: float, angle: float, oil_area: float, area: float,
             tank_status: str, oil_status: int, description: str):
    plt.clf()

    p0, p1, p2, p3 = tank_points
    plt.plot([p0.x, p1.x, p2.x, p3.x, p0.x], [p0.y, p1.y, p2.y, p3.y, p0.y])
    plt.annotate("p0", [p0.x, p0.y])
    plt.annotate("p1", [p1.x, p1.y])
    plt.annotate("p2", [p2.x, p2.y])
    plt.annotate("p3", [p3.x, p3.y])

    (b0, b1, b2), (m0, m1, m2, m3), (t0, t1, t2) = bottom_triangle, middle_parallelogram, top_triangle
    plt.fill_between([b1.x, b0.x, b2.x], [b1.y, b0.y, b2.y], b1.y, color="red", alpha=0.2)
    plt.fill_between([t0.x, t2.x, t1.x], t0.y, [t0.y, t2.y, t1.y], color="blue", alpha=0.2)
    if m0.x <= m3.x:
        plt.fill_between([m0.x, m3.x, m1.x, m2.x], [m0.y, m0.y, m0.y, m2.y], [m0.y, m2.y, m2.y, m2.y],
                         color="green", alpha=0.2)
    else:
        plt.fill_between([m3.x, m0.x, m2.x, m1.x], [m3.y, m1.y, m1.y, m1.y], [m3.y, m3.y, m3.y, m1.y],
                         color="green", alpha=0.2)

    plt.scatter(barycenter.x, barycenter.y, s=100, c="red")

    plt.plot([o1.x, o2.x], [o1.y, o2.y], linewidth=3)
    plt.annotate("o1", [o1.x, o1.y])
    plt.annotate("o2", [o2.x, o2.y])

    plt.title(f"{description}\n"
              f"length = {length:.2f}, height = {height:.2f}, angle = {math.degrees(angle):.2f}, \n"
              f"tank_status = {tank_status}, oil_status = {oil_status}, \n"
              f"oil_area = {oil_area:.2f}, area = {area:.2f}, \n"
              f"barycenter = ({barycenter.x:.2f}, {barycenter.y:.2f})")

    plt.axis('equal')
    plt.show()


def _calc_2d_barycenter(length: float, height: float, angle: float, oil_area: float, description: str) -> Point2d:
    p0 = Point2d(0, 0)
    p1 = Point2d(length * math.cos(angle), length * math.sin(angle))
    p3 = Point2d(height * math.cos(math.pi / 2 + angle), height * math.sin(math.pi / 2 + angle))
    p2 = p1 + p3
    tank_status, bottom_triangle, middle_parallelogram, top_triangle = _tank_status_and_control_points(p0, p1, p2, p3)
    oil_status, barycenter, area, o1, o2 = _oil_status_and_barycenter(bottom_triangle, middle_parallelogram,
                                                                      top_triangle, oil_area)
    # _plot_2d((p0, p1, p2, p3), bottom_triangle, middle_parallelogram, top_triangle,
    #          barycenter, o1, o2, length, height, angle, oil_area, area, tank_status, oil_status, description)
    if math.fabs(oil_area - area) > 1e-6:
        raise ValueError("oil_area must be equal to area")
    return barycenter


def _calc_3d_barycenter(oil_tank_middle_position: Point3d, oil_tank_size: Point3d,
                        oil_volume: float, angle: float, description: str) -> Tuple[Point3d, float]:
    barycenter2d_tank = _calc_2d_barycenter(oil_tank_size.x, oil_tank_size.z, angle, oil_volume / oil_tank_size.y,
                                            description)
    barycenter2d_aircraft = Point2d(
        math.cos(-angle) * barycenter2d_tank.x - math.sin(-angle) * barycenter2d_tank.y,
        math.sin(-angle) * barycenter2d_tank.x + math.cos(-angle) * barycenter2d_tank.y
    )
    barycenter3d_x = oil_tank_middle_position.x - oil_tank_size.x / 2 + barycenter2d_aircraft.x
    barycenter3d_y = oil_tank_middle_position.y
    barycenter3d_z = oil_tank_middle_position.z - oil_tank_size.z / 2 + barycenter2d_aircraft.y
    barycenter3d = Point3d(barycenter3d_x, barycenter3d_y, barycenter3d_z)
    mass = oil_volume * OIL_DENSITY_KGpm3
    return barycenter3d, mass


def calc_3d_barycenter_all_tanks(oil_tank_middle_position_list: List[Point3d], oil_tank_size_list: List[Point3d],
                                 oil_volume_list: List[float], angle: float, description: str) -> Tuple[Point3d, float]:
    composed_mass = 0.0
    composed_barycenter = Point3d(0.0, 0.0, 0.0)
    for k, (oil_tank_middle_position, oil_tank_size, oil_volume) in \
            enumerate(zip(oil_tank_middle_position_list, oil_tank_size_list, oil_volume_list), start=1):
        description_k = f"{description}, No.{k}-OilTank"
        barycenter, mass = _calc_3d_barycenter(oil_tank_middle_position, oil_tank_size, oil_volume, angle,
                                               description_k)
        composed_barycenter = composed_barycenter + mass * barycenter
        composed_mass = composed_mass + mass
    composed_barycenter = composed_barycenter / composed_mass
    return composed_barycenter, composed_mass


if __name__ == '__main__':
    _calc_2d_barycenter(1.0, 2.0, math.radians(-45.0), 0.00, "")
    _calc_2d_barycenter(1.0, 2.0, math.radians(-45.0), 0.50, "")
    _calc_2d_barycenter(1.0, 2.0, math.radians(-45.0), 1.00, "")
    _calc_2d_barycenter(1.0, 2.0, math.radians(-45.0), 1.50, "")
    _calc_2d_barycenter(1.0, 2.0, math.radians(-45.0), 2.00, "")
