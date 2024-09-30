import asyncio
import time
import numpy as np
import pyetp.resqml_objects as ro
from pyetp.types import DataArrayIdentifier
from resqpy.property.property_collection import PropertyCollection
from typing import Tuple
import resqpy.property as rqp
import resqpy.model as rq
import resqpy.unstructured as rug
import resqpy.crs as rqc
import resqpy.time_series as rts
import xmltodict
from warmth_utils.geomint import model_spec
from warmth_utils.osdu import get_mesh_spec, get_simulation_id
from warmth_utils.config import config, MESH_PATH
from warmth_utils.auth import msal_token
import typing
from pyetp import connect
from pyetp.uri import DataObjectURI, DataspaceURI


# Helper functions
async def put_resqml_objects(obj: ro.AbstractObject):
    async with connect(msal_token()) as client:
        rddms_out = await client.put_resqml_objects(obj, dataspace=config.RDDMSDataspace)
        return rddms_out

async def put_data_array(cprop0: ro.AbstractObject, data: np.ndarray, url_epc: typing.Union[DataspaceURI,str]):
    assert isinstance(cprop0, ro.ContinuousProperty) or isinstance(cprop0, ro.DiscreteProperty), "prop must be a Property"
    assert len(cprop0.patch_of_values) == 1, "property obj must have exactly one patch of values"
    async with connect(msal_token()) as client:
        response = await client.put_array(
            DataArrayIdentifier(
                uri=url_epc.raw_uri if isinstance(url_epc, DataObjectURI) else url_epc,
                pathInResource=cprop0.patch_of_values[0].values.values.path_in_hdf_file,
            ),
            data,  # type: ignore
        )
        return response 
    
async def get_resqml_object(url):
    async with connect(msal_token()) as client:
        rddms_out = await client.get_data_objects(url)
        return rddms_out
def store_time_series_data(pc: PropertyCollection, gts, data, props: ro.ContinuousProperty | ro.DiscreteProperty, source_info: str):
    print(f"Add timeseries data {props.citation.title} to model")
    # nodes0 = nodes.copy()
    if isinstance(props, ro.DiscreteProperty):
        discrete = True
    else:
        discrete = False
    points = False
    if props.facet[0].value == "points":
        points = True  # allow  2d array with x y z
    if isinstance(props.property_kind, ro.StandardPropertyKind):
        prop_kind = str(props.property_kind.kind).split(
            ".")[-1].lower().replace("_", " ")
        uom = str(props.uom).split(".")[-1].lower()
        for time_index in range(data.shape[0]-1, -1, -1):  # oldest first
            # last index in resqml is youngest

            data_this_timestep = data[time_index].astype(np.float32)

            pc.add_cached_array_to_imported_list(data_this_timestep,
                                                 source_info,
                                                 props.citation.title,
                                                 uom=uom,
                                                 property_kind=prop_kind,
                                                 realization=0,
                                                 time_index=time_index,
                                                 indexable_element=str(
                                                     props.indexable_element).split(".")[-1].lower(),
                                                 points=points,
                                                 discrete=discrete)
        pc.write_hdf5_for_imported_list()
        pc.create_xml_for_imported_list_and_add_parts_to_model(
            time_series_uuid=gts.uuid)
    else:
        if source_info == 'Vitrinite reflectance':
            kw = "%Ro"
            uom = 'percent'
            property_kind = 'dimensionless'
        if source_info.startswith("Transformation"):
            kw = source_info
            uom = 'percent'
            property_kind = 'dimensionless'
        for time_index in range(data.shape[0]-1, -1, -1):  # oldest first
            # last index in resqml is youngest

            data_this_timestep = data[time_index].astype(np.float32)
            pc.add_cached_array_to_imported_list(data_this_timestep,
                                                 source_info,
                                                 kw,
                                                 uom=uom,
                                                 property_kind=property_kind,
                                                 realization=0,
                                                 time_index=time_index,
                                                 indexable_element=str(props.indexable_element).split(".")[-1].lower())
    return
async def timeseries_prop_fetch(epc:str,input_horizons_ages:list[int],times_in_years_original:list[int],props: list[str], shape):
    points = np.zeros(shape)
    total_download = len(input_horizons_ages)
    count = 0
    for props_index in range(len(props)):
        url_points = props[props_index]
        props1: ro.ContinuousProperty = await get_mesh_prop_meta(url_points)
        # store according to time index
        time_of_props = times_in_years_original[props1.time_index.index]
        try:
            index_in_new_gts = input_horizons_ages.index(time_of_props)
        except ValueError:
            continue
        arr = await get_mesh_prop_array( epc, props1)
        points[index_in_new_gts] = arr
        count += 1
    assert count == total_download
    return points, props1

async def fetch_and_save_timeseries_data(epc:str, input_horizons_ages:list[int],times_in_years_original:list[int],pc, gts, uri: list[str], source_info: str, data_shape: Tuple[int, int]):
    data, props = await timeseries_prop_fetch(epc, input_horizons_ages, times_in_years_original, uri, data_shape)
    store_time_series_data(pc, gts, data, props, source_info)
    return

def parse_age(data):
    return data


def store_non_timeseries_data(resqpyModel: rq.Model, props: ro.ContinuousProperty | ro.DiscreteProperty, hexa_uuid, data: np.ndarray):
    if isinstance(props, ro.DiscreteProperty):
        discrete = True
        uom = 'Euc'
    else:
        discrete = False
        uom = str(props.uom).split(".")[-1].lower()
    if isinstance(props.property_kind, ro.LocalPropertyKind):
        prop_kind = str(
            props.property_kind.local_property_kind.title).replace("_", " ")
    elif isinstance(props.property_kind, ro.StandardPropertyKind):
        prop_kind = str(props.property_kind.kind).split(
            ".")[-1].lower().replace("_", " ")
    else:
        raise Exception("Handle local props kind")
    if discrete:
        data = data.astype(np.int32)
    _ = rqp.Property.from_array(resqpyModel,
                                data,
                                source_info='SubsHeat',
                                keyword=props.citation.title,
                                support_uuid=hexa_uuid,
                                property_kind=prop_kind,
                                indexable_element=str(
                                    props.indexable_element).split(".")[-1].lower(),
                                uom=uom,
                                discrete=discrete)

async def fetch_store_non_timeseries(resqpyModel: rq.Model,mesh_epc, data_url, hexa_uuid, prop_name: str):
    props1, values1 = await get_mesh_property(mesh_epc, data_url)
    if prop_name == "Age":
        values1 = parse_age(values1)
    store_non_timeseries_data(resqpyModel,props1, hexa_uuid, values1)
    return
async def timeseries_prop_nodes(props:list[str], mesh_epc, shape):
    points = np.zeros(shape)
    for time_index in range(shape[0]):
        url_points = props[time_index]
        props1, values1 = await get_mesh_property(mesh_epc, url_points)
        points[props1.time_index.index,:] = values1
    return points

async def get_mesh_prop_array(epc_uri, cprop):
    sem = asyncio.Semaphore(config.N_THREADS)
    async with sem:
        async with connect(msal_token()) as client:
            return await client.get_array(
                DataArrayIdentifier(
                    uri=str(epc_uri), pathInResource=cprop.patch_of_values[0].values.values.path_in_hdf_file,
                )
            )

async def get_mesh_points(epc_uri, uns_uri):
    sem = asyncio.Semaphore(config.N_THREADS)
    async with sem:
        async with connect(msal_token()) as client:
            uns, = await client.get_resqml_objects(uns_uri)

            nodes_per_face = await client.get_array(
                DataArrayIdentifier(
                    uri=str(epc_uri), pathInResource=uns.geometry.nodes_per_face.elements.values.path_in_hdf_file
                )
            )

            points = await client.get_array(
                DataArrayIdentifier(
                    uri=str(epc_uri), pathInResource=uns.geometry.points.coordinates.path_in_hdf_file
                )
            )

            faces_per_cell = await client.get_array(
                DataArrayIdentifier(
                    uri=str(epc_uri), pathInResource=uns.geometry.faces_per_cell.elements.values.path_in_hdf_file
                )
            )

            cell_face_is_right_handed = await client.get_array(
                DataArrayIdentifier(
                    uri=str(epc_uri), pathInResource=uns.geometry.cell_face_is_right_handed.values.path_in_hdf_file
                )
            )
            return uns, points, nodes_per_face, faces_per_cell, cell_face_is_right_handed


async def get_mesh_property(epc_uri, prop_uri):
    sem = asyncio.Semaphore(config.N_THREADS)
    async with sem:
        async with connect(msal_token()) as client:
            rddms_out = await client.get_epc_mesh_property(epc_uri, prop_uri)
            return rddms_out
        
async def get_mesh_prop_meta(prop_uri):
    async with connect(msal_token()) as client:
        cprop0, = await client.get_resqml_objects(prop_uri)
    return cprop0

async def get_mesh_arr_metadata(epc_uri, prop_uri):
    sem = asyncio.Semaphore(config.N_THREADS)
    async with sem:
        async with connect(msal_token()) as client:
            cprop0, = await client.get_resqml_objects(prop_uri)

            # some checks
            assert isinstance(cprop0, ro.ContinuousProperty) or isinstance(cprop0, ro.DiscreteProperty), "prop must be a Property"
            assert len(cprop0.patch_of_values) == 1, "property obj must have exactly one patch of values"

            # # get array
            uid = DataArrayIdentifier(
                    uri=str(epc_uri), pathInResource=cprop0.patch_of_values[0].values.values.path_in_hdf_file,
                )
            return await client.get_array_metadata(uid)

async def download_epc():
    sim_id = get_simulation_id(model_spec.model.id, model_spec.model.version)
    mesh_dict = get_mesh_spec(sim_id)


    input_horizons_ages = [int(
        i.age*-1e6) for i in reversed(model_spec.model.framework.geometries)]

    mesh_uri = mesh_dict["mesh"]
    props_uri = mesh_dict["properties"]
    mesh_epc: str = [i for i in mesh_uri if "EpcExternalPartReference" in i][0]
    # mesh_crs = [i for i in mesh_uri if "LocalDepth3dCrs" in i][0]
    mesh_uns: str = [i for i in mesh_uri if "UnstructuredGridRepresentation" in i][0]
    mesh_ts:str = [i for i in mesh_uri if "TimeSeries" in i][0]
    model = rq.new_model(str(MESH_PATH))
    crs = rqc.Crs(model)
    crs.create_xml()

    # time series
    timeseries_object = await get_resqml_object(mesh_ts)
    timeseries_object = xmltodict.parse(timeseries_object[0].data)
    times_in_years_original = [int(
        i["ns0:YearOffset"]) for i in timeseries_object["ns0:TimeSeries"]["ns0:Time"]]
    # new gts with only input horizon age
    gts = rts.GeologicTimeSeries.from_year_list(
        model, input_horizons_ages, title="warmth simulation")
    gts.create_xml()
    rts.timeframe_for_time_series_uuid(model, gts.uuid)
    # mesh
    # create an empty HexaGrid
    uns, points, nodes_per_face, faces_per_cell, cell_face_is_right_handed = await get_mesh_points(mesh_epc, mesh_uns)

    num_time_indices = len(input_horizons_ages)


    hexa = rug.HexaGrid(model, title="hexamesh")
    assert hexa.cell_shape == 'hexahedral'

    hexa.crs_uuid = model.uuid(obj_type='LocalDepth3dCrs')

    hexa.set_cell_count(uns.cell_count)
    # faces
    hexa.face_count = uns.geometry.face_count
    hexa.faces_per_cell_cl = np.arange(6, 6 * uns.cell_count + 1, 6, dtype=int)
    hexa.faces_per_cell = np.array(faces_per_cell)

    # nodes
    hexa.node_count = uns.geometry.node_count
    hexa.nodes_per_face_cl = np.arange(
        4, 4 * uns.geometry.face_count + 1, 4, dtype=int)
    hexa.nodes_per_face = np.array(nodes_per_face)
    hexa.cell_face_is_right_handed = cell_face_is_right_handed

    # points
    hexa.points_cached = points  # nodes_loc[0,:,:]

    # basic validity check
    hexa.check_hexahedral()

    hexa.create_xml()
    hexa.write_hdf5()

    if hexa.property_collection is None:
        hexa.property_collection = rqp.PropertyCollection(support=hexa)

    pc = hexa.property_collection

    # properties

    temperature_time_series_data_shape = (
        num_time_indices, uns.geometry.node_count)
    cell_time_series_data_shape = (num_time_indices, uns.cell_count)
    timeseries_data = [[props_uri['Temperature'][1], "Temperature",
                        temperature_time_series_data_shape],
                       [props_uri['%Ro'][1], "Vitrinite reflectance",
                           temperature_time_series_data_shape],

                       [props_uri['Transformation_ratio_gas'][1],
                        'Transformation_ratio_gas', cell_time_series_data_shape],
                       [props_uri['Transformation_ratio_oil'][1],
                        'Transformation_ratio_oil', cell_time_series_data_shape],
                       [props_uri['Transformation_ratio_oil2gas'][1],
                        'Transformation_ratio_oil2gas', cell_time_series_data_shape],
                       [props_uri['Expelled_oil_specific'][1],
                        'fluid volume', cell_time_series_data_shape],
                       [props_uri['Expelled_gas_specific'][1], 'fluid volume', cell_time_series_data_shape]]

    for i in timeseries_data:
        try:
            await fetch_and_save_timeseries_data(mesh_epc,input_horizons_ages,times_in_years_original, pc, gts, i[0], i[1], i[2])
        except:
            time.sleep(60)
            await fetch_and_save_timeseries_data(mesh_epc,input_horizons_ages,times_in_years_original,pc, gts, i[0], i[1], i[2])
        time.sleep(10)

    non_timeseries_data = ["Porosity_decay", "LayerID", "Radiogenic_heat_production",
                           "Age", "Porosity_initial", "thermal_conductivity"]

    for i in non_timeseries_data:
        assert len(props_uri[i][1]) == 1
        await fetch_store_non_timeseries(model, mesh_epc, props_uri[i][1][0], hexa.uuid, i)

    model.store_epc()

    # TEST READ LOCAL FILE
    m = rq.Model(str(MESH_PATH))
    assert m is not None
    ts_uuid_2 = m.uuid(obj_type='GeologicTimeSeries')
    uuids = m.uuids(obj_type='ContinuousProperty')
    prop_titles = [rqp.Property(m, uuid=u).title for u in uuids]
    uuids = m.uuids(obj_type='DiscreteProperty')
    prop_titles = [rqp.Property(m, uuid=u).title for u in uuids]
    return MESH_PATH


