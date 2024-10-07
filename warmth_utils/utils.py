from shapely import Point, Polygon

from warmth_utils.spec import TectonicModel


def point_in_poly(point: list[float], polygon: list[list[float]]) -> tuple[bool, float]:
    point = Point(point)
    polygon = Polygon(polygon)
    return polygon.contains(point), point.distance(polygon)

def find_tectonic_model_index(point: list[float], tectonic: TectonicModel) -> int:
    all_distance = []
    for idx, model in enumerate(tectonic.domains):
        within_polygon, distance = point_in_poly(point, model.extent)
        if within_polygon:
            return idx
        all_distance.append(distance)
    return all_distance.index(min(all_distance))