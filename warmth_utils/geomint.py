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
