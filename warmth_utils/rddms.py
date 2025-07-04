import asyncio
import logging
import time
from uuid import UUID
import numpy as np
from scipy.interpolate import griddata
from pathlib import Path
import pyetp.resqml_objects as ro
from pyetp.types import DataArrayIdentifier
from resqpy.property.property_collection import PropertyCollection
from typing import Literal, Tuple
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


def dataspace_uri() -> DataspaceURI:
    sim_id = get_simulation_id()
    sim_id_short = sim_id.split(":")[-1]
    return DataspaceURI.from_name(f"{config.SIMULATIONDATASPACEPREFIX}/{sim_id_short}")

async def get_map_value(rddms: list[str],x:float,y:float,sampling: Literal["linear","nearest"])-> float:
    epc_url =  [i for i in rddms if "EpcExternalPartReference" in i][0]
    gri_url =  [i for i in rddms if "Grid2dRepresentation" in i][0]
    crs_url =  [i for i in rddms if "LocalDepth3dCrs" in i]
    async with connect(msal_token()) as client:
        v = await client.get_surface_value_x_y(epc_url, gri_url, crs_url, x,y,sampling)
    return v

async def download_map(epc_uri, gri_uri, crs_uri, save_path:str):
    async with connect(msal_token()) as client:
        surf = await client.get_xtgeo_surface(epc_uri,gri_uri,crs_uri)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        surf.to_file(save_path)
        return
    

async def put_resqml_objects(obj: ro.AbstractObject):
    ds_uri = dataspace_uri()
    async with connect(msal_token()) as client:
        rddms_out = await client.put_resqml_objects(obj, dataspace_uri=ds_uri)
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
    # Supress transmissivity warning
    logging.getLogger("resqpy.property._collection_add_part").setLevel(logging.ERROR)
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
    logging.info(f"Added timeseries data {props.citation.title} to model")
    return

async def timeseries_prop_fetch(epc:str,input_horizons_ages:list[int],times_in_years_original:list[int],props: list[str], shape):
    points = np.zeros(shape)
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
        logging.debug(f"fetch {props1.citation.title} progress {count}/{points.shape[0]} ")
    logging.info(f"fetched {props1.citation.title} for all {points.shape[0]} timesteps")
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
    logging.info(f"Added non-timeseries data {props.citation.title} to model")
    return

async def fetch_store_non_timeseries(resqpyModel: rq.Model,mesh_epc, data_url, hexa_uuid, prop_name: str):
    props1, values1 = await get_mesh_property(mesh_epc, data_url)
    if prop_name == "Age":
        values1 = parse_age(values1)
    store_non_timeseries_data(resqpyModel,props1, hexa_uuid, values1)
    return
def fill_nan_interp1d(arr):
    assert len(arr.shape) == 1
    non_nan = ~np.isnan(arr)
    xp = non_nan.ravel().nonzero()[0]
    fp = arr[~np.isnan(arr)]
    x  = np.isnan(arr).ravel().nonzero()[0]
    arr[np.isnan(arr)] = np.interp(x, xp, fp)
    return arr

async def timeseries_prop_nodes(props:list[str], mesh_epc, shape):
    assert len(shape) > 1 and len(shape) < 4
    buffer = np.full(shape, fill_value=np.nan)
    async def populate(url_points):
        props1, values1 = await get_mesh_property(mesh_epc, url_points)
        buffer[props1.time_index.index,:] = values1
        return
    await asyncio.gather(*[populate(url_points)for url_points in props])
    if len(shape) == 2: # temperature
        # first dimension is n_time, second is n_node # for temperature
        for idx in range (buffer.shape[1]):
            v = buffer[:, idx]
            buffer[:, idx]  = fill_nan_interp1d(v)
    else: # points
        for idx in range (buffer.shape[1]):
            arr = buffer[:,idx, :]
            arr[:,:2] = arr[0,:2]
            z_val = arr[:,-1]
            arr[:,-1] = fill_nan_interp1d(z_val)
            buffer[:,idx, :] = arr
    return buffer

async def get_mesh_prop_array(epc_uri, cprop):
    sem = asyncio.Semaphore(config.N_THREADS)
    async with sem:
        try:
            async with connect(msal_token()) as client:
                return await client.get_array(
                    DataArrayIdentifier(
                        uri=str(epc_uri), pathInResource=cprop.patch_of_values[0].values.values.path_in_hdf_file,
                    )
                )
        except:
            async with connect(msal_token()) as client:
                return await client.get_array(
                    DataArrayIdentifier(
                        uri=str(epc_uri), pathInResource=cprop.patch_of_values[0].values.values.path_in_hdf_file,
                    )
                )
            
async def get_mesh_points_only(epc_uri, uns_uri):
    sem = asyncio.Semaphore(config.N_THREADS)
    async with sem:
        try:
            async with connect(msal_token()) as client:
                uns, = await client.get_resqml_objects(uns_uri)
        except:
            async with connect(msal_token()) as client:
                uns, = await client.get_resqml_objects(uns_uri)
        try:
            async with connect(msal_token()) as client:
                points = await client.get_array(
                    DataArrayIdentifier(
                        uri=str(epc_uri), pathInResource=uns.geometry.points.coordinates.path_in_hdf_file
                    )
                )
        except:
            async with connect(msal_token()) as client:
                points = await client.get_array(
                    DataArrayIdentifier(
                        uri=str(epc_uri), pathInResource=uns.geometry.points.coordinates.path_in_hdf_file
                    )
                )

        return uns, points
    
async def get_mesh_points(epc_uri, uns_uri):
    sem = asyncio.Semaphore(config.N_THREADS)
    async with sem:
        uns, points = await get_mesh_points_only(epc_uri, uns_uri)
        try:
            async with connect(msal_token()) as client:
                nodes_per_face = await client.get_array(
                    DataArrayIdentifier(
                        uri=str(epc_uri), pathInResource=uns.geometry.nodes_per_face.elements.values.path_in_hdf_file
                    )
                )
        except:
            async with connect(msal_token()) as client:
                nodes_per_face = await client.get_array(
                    DataArrayIdentifier(
                        uri=str(epc_uri), pathInResource=uns.geometry.nodes_per_face.elements.values.path_in_hdf_file
                    )
                )
        try:
            async with connect(msal_token()) as client:
                faces_per_cell = await client.get_array(
                    DataArrayIdentifier(
                        uri=str(epc_uri), pathInResource=uns.geometry.faces_per_cell.elements.values.path_in_hdf_file
                    )
                )
        except:
            async with connect(msal_token()) as client:
                faces_per_cell = await client.get_array(
                    DataArrayIdentifier(
                        uri=str(epc_uri), pathInResource=uns.geometry.faces_per_cell.elements.values.path_in_hdf_file
                    )
                    )
        try:
            async with connect(msal_token()) as client:
                cell_face_is_right_handed = await client.get_array(
                    DataArrayIdentifier(
                        uri=str(epc_uri), pathInResource=uns.geometry.cell_face_is_right_handed.values.path_in_hdf_file
                    )
                )
        except:
            async with connect(msal_token()) as client:
                cell_face_is_right_handed = await client.get_array(
                    DataArrayIdentifier(
                        uri=str(epc_uri), pathInResource=uns.geometry.cell_face_is_right_handed.values.path_in_hdf_file
                    )
                )
        return uns, points, nodes_per_face, faces_per_cell, cell_face_is_right_handed


async def get_mesh_property(epc_uri, prop_uri):
    async with connect(msal_token()) as client:
        rddms_out = await client.get_epc_mesh_property(epc_uri, prop_uri)
        return rddms_out
        
async def get_mesh_prop_meta(prop_uri):
    async with connect(msal_token()) as client:
        cprop0, = await client.get_resqml_objects(prop_uri)
    return cprop0

async def get_mesh_arr_metadata(epc_uri, prop_uri):
    async with connect(msal_token()) as client:
        cprop0, = await client.get_resqml_objects(prop_uri)
        uid = DataArrayIdentifier(
                uri=str(epc_uri), pathInResource=cprop0.patch_of_values[0].values.values.path_in_hdf_file,
            )
        return await client.get_array_metadata(uid)


async def download_epc():
    sim_id = get_simulation_id()
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
    logging.debug("Got timeseries object")
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
    async with connect(msal_token()) as client:
        uns, = await client.get_resqml_objects(mesh_uns)
    logging.debug("Got uns object")
    async with connect(msal_token()) as client:
        points = await client.get_array(
            DataArrayIdentifier(
                uri=str(mesh_epc), pathInResource=uns.geometry.points.coordinates.path_in_hdf_file
            )
        )
    assert points.shape[0] == uns.geometry.node_count
    logging.debug("Got points")
    async with connect(msal_token()) as client:
        nodes_per_face = await client.get_array(
            DataArrayIdentifier(
                uri=str(mesh_epc), pathInResource=uns.geometry.nodes_per_face.elements.values.path_in_hdf_file
            )
        )
    logging.debug("Got nodes per face")
    async with connect(msal_token()) as client:
        faces_per_cell = await client.get_array(
            DataArrayIdentifier(
                uri=str(mesh_epc), pathInResource=uns.geometry.faces_per_cell.elements.values.path_in_hdf_file
            )
        )
    logging.debug("Got face per cell")
    async with connect(msal_token()) as client:
        cell_face_is_right_handed = await client.get_array(
            DataArrayIdentifier(
                uri=str(mesh_epc), pathInResource=uns.geometry.cell_face_is_right_handed.values.path_in_hdf_file
            )
        )
    logging.debug("Got cell faces right handed")
    #uns, points, nodes_per_face, faces_per_cell, cell_face_is_right_handed = await get_mesh_points(mesh_epc, mesh_uns)
    logging.info("Got all mesh data")
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
    logging.getLogger("resqpy").setLevel(logging.ERROR)  
    for i in timeseries_data:
        try:
            await fetch_and_save_timeseries_data(mesh_epc,input_horizons_ages,times_in_years_original, pc, gts, i[0], i[1], i[2])
        except:
            time.sleep(60)
            await fetch_and_save_timeseries_data(mesh_epc,input_horizons_ages,times_in_years_original,pc, gts, i[0], i[1], i[2])
        time.sleep(10)
    logging.getLogger("resqpy").setLevel(logging.WARNING) 
    non_timeseries_data = ["Porosity_decay", "LayerID", "Radiogenic_heat_production",
                           "Age", "Porosity_initial", "thermal_conductivity"]
    for i in non_timeseries_data:
        assert len(props_uri[i][1]) == 1
        await fetch_store_non_timeseries(model, mesh_epc, props_uri[i][1][0], hexa.uuid, i)

    model.store_epc()

    # TEST READ LOCAL FILE
    # m = rq.Model(str(MESH_PATH))
    # assert m is not None
    # ts_uuid_2 = m.uuid(obj_type='GeologicTimeSeries')
    # uuids = m.uuids(obj_type='ContinuousProperty')
    # prop_titles = [rqp.Property(m, uuid=u).title for u in uuids]
    # uuids = m.uuids(obj_type='DiscreteProperty')
    # prop_titles = [rqp.Property(m, uuid=u).title for u in uuids]
    return MESH_PATH


