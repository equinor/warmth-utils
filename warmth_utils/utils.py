from shapely import Point, Polygon


def point_in_poly(point: list[float], polygon: list[list[float]]) -> tuple[bool, float]:
    point = Point(point)
    polygon = Polygon(polygon)
    return polygon.contains(point), point.distance(polygon)