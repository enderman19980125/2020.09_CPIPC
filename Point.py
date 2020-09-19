import math


class Point2d:
    def __init__(self, x: float, y: float):
        self.x = 1.0 * x
        self.y = 1.0 * y

    def __add__(self, other):
        return Point2d(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point2d(self.x - other.x, self.y - other.y)

    def __rmul__(self, other):
        return Point2d(other * self.x, other * self.y)

    def __truediv__(self, other):
        return Point2d(self.x / other, self.y / other)

    def __str__(self):
        return f"({self.x:.3f}, {self.y:.3f})"


class Point3d:
    def __init__(self, x: float, y: float, z: float):
        self.x = 1.0 * x
        self.y = 1.0 * y
        self.z = 1.0 * z

    def __add__(self, other):
        return Point3d(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Point3d(self.x - other.x, self.y - other.y, self.z - other.z)

    def __rmul__(self, other):
        return Point3d(other * self.x, other * self.y, other * self.z)

    def __truediv__(self, other):
        return Point3d(self.x / other, self.y / other, self.z / other)

    def __str__(self):
        return f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


def distance3d(p1: Point3d, p2: Point3d) -> float:
    dis = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
    return dis


if __name__ == '__main__':
    p1 = Point3d(2, 2, 2)
    p2 = Point3d(1, 1, 1)
    p3 = p1 - p2
    print(p3)
