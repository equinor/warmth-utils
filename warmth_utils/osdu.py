from pathlib import Path
from azure.storage.blob import BlobClient
import time
import requests
import pyetp.resqml_objects as ro
from warmth_utils.config import config
from warmth_utils.auth import msal_token



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
    if r.status_code != 200:
        print(r.text)
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


def get_simulation_id(model_id:str, version:int)-> str:
    model_obj_osdu_id = fr"{config.OSDUPARTITION}\:work-product-component--Activity\:{model_id}\:{int(version)}"
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
    if r.status_code != 200:
        print(r.text)
    r = r.json()
    return r["results"][0]["id"]

def get_obj(obj_id:str):
    uri = f"{config.OSDUHOST}/api/storage/v2/records/{obj_id}"
    r = requests.get(url=uri,headers=get_header())
    if r.status_code != 200:
        print(r.text)
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
        print(r.text)
        raise Exception("Cannot update new mesh")      
    return


    
def get_file_uploadURL():
    r = requests.get(f"{config.OSDUHOST}/api/file/v2/files/uploadURL", headers=get_header(),params={"expiryTime":"15M"})
    if r.status_code != 200:
        print(r.text)
    return r.json()
#)

def upload_file(SignedURL: str, filepath:str| Path):
    blob_service_client = BlobClient.from_blob_url(SignedURL)
    with open(filepath, "rb") as f:
        blob_service_client.upload_blob(data=f,overwrite=True)
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
            }
        },
        "acl":meshObj["acl"]
        }
    uri = f"{config.OSDUHOST}/api/file/v2/files/metadata"
    r = requests.post(url=uri,headers=get_header(),json=data)
    if r.status_code != 200:
        print(r.text)
    return r.json()["id"]

def add_migriResults_to_mesh_manifest(fileObjId:str, existingMeshObj:dict, result_maps:dict):
    existingMeshObj["data"]["ExtensionProperties"]["migris"] = result_maps
    existingMeshObj["data"]["Datasets"]= [fileObjId]
    return overwrite_mesh_obj(existingMeshObj)

def upload_migri_results(model_id: str,model_version:int, filepath: str| Path, result_maps: dict):
    sim_id = get_simulation_id(model_id, model_version)
    signedURL = get_file_uploadURL()
    upload_file(signedURL["Location"]["SignedURL"], filepath)
    meshObj = get_mesh_manifest(sim_id)
    fileObj_id= put_file_manifest(meshObj,signedURL["Location"]["FileSource"])
    add_migriResults_to_mesh_manifest(fileObj_id, meshObj, result_maps)
    return