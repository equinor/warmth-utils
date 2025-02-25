import logging
from pathlib import Path
from azure.storage.blob import BlobClient
import time
import requests
import pyetp.resqml_objects as ro
from warmth_utils.config import config
from warmth_utils.auth import msal_token
from warmth_utils.geomint import model_spec


def searchMesh(sim_id:str):
    query = fr'data.LineageAssertions.ID:{config.OSDUPARTITION}\:work-product-component--Activity\:{sim_id}\:'
    uri = f"{config.OSDUHOST}/api/search/v2/query"
    data = {
        "kind": f"osdu:wks:work-product-component--UnstructuredGridRepresentation:{config.MESHMANIFESTVERSION}",
        "query": query,
        "returnedFields": ['id'],
        "sort": {
          "field": [
            'createTime'
          ],
          "order": [
            'DESC'
          ]
        }
      }
    r = requests.post(uri, headers= get_header(),json=data)
    try:
        r.raise_for_status()
    except Exception as e:
        logging.error(f"Failed search mesh manifest {r.text}")
        raise e
    return r.json()
def get_mesh_spec(sim_id:str)-> dict:
    data = get_mesh_manifest(sim_id)
    return data["data"]["ExtensionProperties"]

def get_mesh_manifest(sim_id:str):
    sim_id = sim_id.split(":")[-1]
    r = searchMesh(sim_id)
    if len(r["results"])==0:
        time.sleep(30)
        r = searchMesh(sim_id)
        mesh_id = r["results"][0]["id"]
    else:
        mesh_id = r["results"][0]["id"]
    return get_obj(mesh_id)

def get_simulation_id(model_id: str, version: int) -> str:
    if isinstance(model_spec.simId, str):
        return model_spec.simId
    max_count = 10
    count = 0
    while count < max_count:
        r = _get_sim_id(model_id,version)
        if len(r) == 0:
            logging.info("Retrying to find sim id")
            time.sleep(5)
            count +=1
        else:
            logging.info(f"Found sim id {r[0]['id']}")
            return r[0]["id"]
    raise Exception(f"Failed to find sim id for {model_id}:{version}")

def _get_sim_id(model_id:str, version:int)-> list:
    version = int(version)
    model_obj_osdu_id = fr"{config.OSDUPARTITION}\:work-product-component--Activity\:{model_id}\:{version}"
    query = f'(tags.geomintType:simulation) AND (data.LineageAssertions.ID:{model_obj_osdu_id})'
    uri = f"{config.OSDUHOST}/api/search/v2/query"
    data = {
        "kind": f"osdu:wks:work-product-component--Activity:{config.ACTIVITYMODELVERSION}",
        "query": query,
        "returnedFields": ['id'],
        "sort": {
          "field": [
            'createTime'
          ],
          "order": [
            'DESC'
          ]
        }
      }
    r = requests.post(uri, headers= get_header(),json=data)
    try:
        r.raise_for_status()
    except Exception as e:
        logging.error(f"Failed getting simulation id {r.text}")
        raise e
    r = r.json()
    return r["results"]

def get_obj(obj_id:str):
    uri = f"{config.OSDUHOST}/api/storage/v2/records/{obj_id}"
    r = requests.get(url=uri,headers=get_header())
    try:
        r.raise_for_status()
    except Exception as e:
        logging.error(f"Failed getting object manifest {r.text}")
        raise e
    return r.json()

def get_header()->dict:
    return {
        'Content-Type': 'application/json',
        'data-partition-id': config.OSDUPARTITION,
        "Authorization": msal_token()
        }


def add_mesh_props_to_meshObj(mesh_obj:dict,extension_prop: dict, additional_rddmsurl: list[str]):
    mesh_obj["data"]["ExtensionProperties"] = extension_prop

    mesh_obj["data"]["DDMSDatasets"].extend(additional_rddmsurl)
    return mesh_obj


def overwrite_mesh_obj (new_mesh_obj:dict) -> None:
    remove = ["version", "createUser", "createTime", "modifyTime", "modifyUser"]
    for i in remove:
        if i in new_mesh_obj:
          del new_mesh_obj[i]
    new_mesh_obj=[new_mesh_obj]
    uri = f"{config.OSDUHOST}/api/storage/v2/records"
    r = requests.put(url=uri, headers=get_header(), json=new_mesh_obj)
    try:
        r.raise_for_status()
    except Exception as e:
        logging.error(f"Failed updating mesh manifest {r.text}")  
        raise e
    return


    
def get_file_uploadURL():
    r = requests.get(f"{config.OSDUHOST}/api/file/v2/files/uploadURL", headers=get_header(),params={"expiryTime":"15M"})
    try:
        r.raise_for_status()
    except Exception as e:
        logging.error(f"Failed getting signed upload url {r.text}")
        raise e
    return r.json()
#)

def upload_file(SignedURL: str, filepath:str| Path):
    logging.getLogger().setLevel(logging.WARNING)
    blob_service_client = BlobClient.from_blob_url(SignedURL)
    with open(filepath, "rb") as f:
        blob_service_client.upload_blob(data=f,overwrite=True)
    logging.getLogger().setLevel(logging.INFO)
    return

def put_file_manifest(meshObj: dict, fileSource: str) -> str:
    data = {
        "kind":f"osdu:wks:dataset--File.Generic:{config.FILEMANIFESTVERSION}",
        "legal":meshObj["legal"],
        "data":{
            "Source":"Migris",
            "DatasetProperties":{
                "FileSourceInfo":{
                    "FileSource":fileSource,
                }
            },
            "TechnicalAssurances": [
                    {
                        "TechnicalAssuranceTypeID": f"{config.OSDUPARTITION}:reference-data--TechnicalAssuranceType:Unevaluated:"
                    }
                ]
        },
        "acl":meshObj["acl"],
        "tags": {"geomintType": "migri"}
        }
    uri = f"{config.OSDUHOST}/api/file/v2/files/metadata"
    r = requests.post(url=uri,headers=get_header(),json=data)
    try:
        r.raise_for_status()
    except Exception as e:
        logging.error(f"Failed creating file manifest {r.text}")
        raise e
    return r.json()["id"]

def add_migriResults_to_mesh_manifest(fileObjId:str, existingMeshObj:dict, result_maps:dict):
    existingMeshObj["data"]["ExtensionProperties"]["migris"] = result_maps
    existingMeshObj["data"]["Datasets"]= [fileObjId]
    return overwrite_mesh_obj(existingMeshObj)

def upload_migri_results(filepath: str| Path, result_maps: dict):
    sim_id = get_simulation_id(model_spec.model.id, model_spec.model.version)
    signedURL = get_file_uploadURL()
    upload_file(signedURL["Location"]["SignedURL"], filepath)
    meshObj = get_mesh_manifest(sim_id)
    fileObj_id= put_file_manifest(meshObj,signedURL["Location"]["FileSource"])
    add_migriResults_to_mesh_manifest(fileObj_id, meshObj, result_maps)
    return