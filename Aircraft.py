import numpy as np
from Point import Point3d

OIL_DENSITY_KGpm3 = 850

BARYCENTER_WITHOUT_OIL = Point3d(0.0, 0.0, 0.0)

OIL_TANK_MIDDLE_POSITION = (
    Point3d(8.91304348, 1.20652174, 0.61669004),
    Point3d(6.91304348, -1.39347826, 0.21669004),
    Point3d(-1.68695652, 1.20652174, -0.28330996),
    Point3d(3.11304348, 0.60652174, -0.18330996),
    Point3d(-5.28695652, -0.29347826, 0.41669004),
    Point3d(-2.08695652, -1.49347826, 0.21669004),
)

OIL_TANK_SIZE = (
    Point3d(1.5, 0.9, 0.3),
    Point3d(2.2, 0.8, 1.1),
    Point3d(2.4, 1.1, 0.9),
    Point3d(1.7, 1.3, 1.2),
    Point3d(2.4, 1.2, 1),
    Point3d(2.4, 1, 0.5),
)

OIL_TANK_INIT_OIL_VOLUME = (0.3, 1.5, 2.1, 1.9, 2.6, 0.8)
OIL_TANK_INIT_OIL_MASS = tuple(np.array(OIL_TANK_INIT_OIL_VOLUME) * OIL_DENSITY_KGpm3)
