import numpy as np
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

RESOLUTION_LOD0: float = 8000
def get_max_zoom(xStep: float|int, yStep: float|int) -> int:
    step = (xStep, yStep)
    return int(np.fmax(0, int(np.log2(RESOLUTION_LOD0 / np.min(step)))) + 1)