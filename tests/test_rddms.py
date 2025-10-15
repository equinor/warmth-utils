import pytest
import numpy as np
import xtgeo
from pyetp import ETPClient
from pyetp.uri import DataspaceURI
import pyetp.resqml_objects as ro
import resqpy.model as rq
import resqpy.property as rqp
import resqpy.unstructured as rug


from warmth_utils.rddms_utils import (
    get_surface_value_x_y,
    put_epc_mesh,
    get_epc_mesh,
    get_epc_mesh_property,
)

def create_surface(ncol: int, nrow: int, rotation: float):
    surface = xtgeo.RegularSurface(
        ncol=ncol,
        nrow=nrow,
        xori=np.random.rand() * 1000,
        yori=np.random.rand() * 1000,
        xinc=np.random.rand() * 1000,
        yinc=np.random.rand() * 1000,
        rotation=rotation,
        values=np.random.random((nrow, ncol)).astype(np.float32),
    )   
    return surface


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "surface", [create_surface(100, 40, 0), create_surface(3, 3, 0)] 
)
async def test_get_xy_from_surface(
    etp_client: ETPClient, surface: xtgeo.RegularSurface, duri: DataspaceURI
):
    # NOTE: xtgeo calls the first axis (axis 0) of the values-array
    # columns, and the second axis by rows.

    epsg_code = 23031
    epc_uri, gri_uri, crs_uri = await etp_client.put_xtgeo_surface(
        surface, epsg_code, duri
    )   
    x_ori = surface.xori
    y_ori = surface.yori
    x_max = x_ori + (surface.xinc * surface.ncol)
    y_max = y_ori + (surface.yinc * surface.nrow)
    x = np.random.uniform(x_ori, x_max)
    y = np.random.uniform(y_ori, y_max)
    nearest = await get_surface_value_x_y(
        etp_client, epc_uri, gri_uri, crs_uri, x, y, "nearest"
    )   
    xtgeo_nearest = surface.get_value_from_xy((x, y), sampling="nearest")
    assert nearest == pytest.approx(xtgeo_nearest)
    linear = await get_surface_value_x_y(
        etp_client, epc_uri, gri_uri, crs_uri, x, y, "bilinear"
    )   
    xtgeo_linear = surface.get_value_from_xy((x, y)) 
    assert linear == pytest.approx(xtgeo_linear)

    # # test x y index fencing
    x_i = x_max - surface.xinc - 1 
    y_i = y_max - surface.yinc - 1 

    linear_i = await get_surface_value_x_y(
        etp_client, epc_uri, gri_uri, crs_uri, x_i, y_i, "bilinear"
    )   
    xtgeo_linear_i = surface.get_value_from_xy((x_i, y_i))
    assert linear_i == pytest.approx(xtgeo_linear_i, rel=1e-2)

    # test outside map coverage
    x_ii = x_max + 100 
    y_ii = y_max + 100 
    linear_ii = await get_surface_value_x_y(
        etp_client, epc_uri, gri_uri, crs_uri, x_ii, y_ii, "bilinear"
    )   
    assert linear_ii is None



@pytest.mark.parametrize(
    "input_mesh_file", ["./data/model_hexa_0.epc", "./data/model_hexa_ts_0_new.epc"]
)
@pytest.mark.asyncio
async def test_mesh(etp_client: ETPClient, duri: DataspaceURI, input_mesh_file: str):
    model = rq.Model(input_mesh_file)
    assert model is not None

    hexa_uuids = model.uuids(obj_type="UnstructuredGridRepresentation")
    hexa = rug.HexaGrid(model, uuid=hexa_uuids[0])
    assert hexa is not None
    assert hexa.nodes_per_face is not None, "hexamesh object is incomplete"
    assert hexa.nodes_per_face_cl is not None, "hexamesh object is incomplete"
    assert hexa.faces_per_cell is not None, "hexamesh object is incomplete"
    assert hexa.faces_per_cell_cl is not None, "hexamesh object is incomplete"
    assert hexa.cell_face_is_right_handed is not None, "hexamesh object is incomplete"

    uuids = model.uuids(obj_type="ContinuousProperty")
    assert len(uuids) == len(set(uuids))

    prop_titles = list(set([rqp.Property(model, uuid=u).title for u in uuids]))
    uuids = model.uuids(obj_type="DiscreteProperty")

    prop_titles = list(
        set(prop_titles + [rqp.Property(model, uuid=u).title for u in uuids])
    )

    # The optional "points" (dynamic nodes) property is neither
    # ContinuousProperty nor DiscreteProperty: special treatment
    node_uuids = model.uuids(title="points")
    special_prop_titles = list(
        set([rqp.Property(model, uuid=u).title for u in node_uuids])
    )
    prop_titles = prop_titles + special_prop_titles
    rddms_uris, prop_uris = await put_epc_mesh(
        etp_client, str(input_mesh_file), hexa.title, prop_titles, 23031, duri
    )

    (
        uns,
        points,
        nodes_per_face,
        nodes_per_face_cl,
        faces_per_cell,
        faces_per_cell_cl,
        cell_face_is_right_handed,
    ) = await get_epc_mesh(etp_client, rddms_uris[0], rddms_uris[2])

    mesh_has_timeseries = len(rddms_uris) > 3 and len(str(rddms_uris[3])) > 0

    assert str(hexa.uuid) == str(uns.uuid), "returned mesh uuid must match"
    np.testing.assert_allclose(points, hexa.points_ref())  # type: ignore

    np.testing.assert_allclose(nodes_per_face, hexa.nodes_per_face)
    np.testing.assert_allclose(nodes_per_face_cl, hexa.nodes_per_face_cl)
    np.testing.assert_allclose(faces_per_cell, hexa.faces_per_cell)
    np.testing.assert_allclose(faces_per_cell_cl, hexa.faces_per_cell_cl)
    np.testing.assert_allclose(
        cell_face_is_right_handed, hexa.cell_face_is_right_handed
    )

    for key, value in prop_uris.items():
        found_indices = set()
        for prop_uri in value[1]:
            prop0, values = await get_epc_mesh_property(
                etp_client, rddms_uris[0], prop_uri
            )
            assert prop0.supporting_representation.uuid == str(uns.uuid), (
                "property support must match the mesh"
            )
            time_index = prop0.time_index.index if prop0.time_index else -1
            assert time_index not in found_indices, f"Duplicate time index {time_index}"
            if mesh_has_timeseries:
                prop_uuids = model.uuids(title=key)
                prop_uuid = prop_uuids[time_index]
            else:
                prop_uuid = model.uuid(title=key)
            prop = rqp.Property(model, uuid=prop_uuid)

            continuous = prop.is_continuous()
            assert isinstance(prop0, ro.ContinuousProperty) == continuous, (
                "property types must match"
            )
            assert isinstance(prop0, ro.DiscreteProperty) == (not continuous), (
                "property types must match"
            )
            np.testing.assert_allclose(
                prop.array_ref(),
                values,
                err_msg=f"property {key} at time_index {time_index} does not match",
            )  # type: ignore
            found_indices.add(time_index)
