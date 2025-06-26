import xtgeo
from warmth_utils.spec import GeomintFullModel
from warmth_utils.config import MODEL_SPEC
import json


with open(MODEL_SPEC, 'r') as f:
    model_spec_dict = json.load(f)
    model_spec = GeomintFullModel.parse_obj(model_spec_dict)
    
pts = []
for p in model_spec.model.aoi:
    p = list(p)
    p.extend([0, 1])
    pts.append(p)
model_xtgeo_polygon = xtgeo.Polygons(pts)

def age_ranges(youngest, oldest, interval):
    r = list(range(youngest, oldest, interval))
    r.append(oldest)
    return r
input_horizons_ages = [int(i.age) for i in model_spec.model.framework.geometries]
rift_events = [i.riftEvents for i in model_spec.model.tectonicModel.domains]
rift_ages = [age_ranges(j.end, j.start, 2)  for i in rift_events for j in i]

output_ages = age_ranges(min(input_horizons_ages), max(input_horizons_ages), 5)

for i in rift_ages:
    output_ages.extend(i)

output_ages = list(set(output_ages))
output_ages.sort()