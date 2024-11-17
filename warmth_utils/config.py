from pathlib import Path
from pyetp.config import SETTINGS
from pyetp.client import logger as pyetp_logger
import logging
from pydantic import BaseSettings, UUID4

class UTIL_SETTINGS(BaseSettings):
    CLIENT_ID: UUID4
    CLIENT_SECRET: str
    TENANT_ID: UUID4
    CACHEDIR: str = "./"
    MODELSPEC:str
    RDDMSURL: str
    RDDMSDATASPACE: str
    OSDUHOST: str
    OSDUPARTITION: str
    OSDURESOURCEID:UUID4
    N_THREADS: int = 4
    
    
    
    ACTIVITYMODELVERSION: str = "1.0.0"
    MESHMANIFESTVERSION:str = "1.1.0"
    FILEMANIFESTVERSION:str = "1.0.0"

config = UTIL_SETTINGS()
CACHE_DIR = Path(config.CACHEDIR)
MESH_PATH = CACHE_DIR / "results.epc"
MODEL_SPEC = Path(config.MODELSPEC)


SETTINGS.etp_url= config.RDDMSURL
SETTINGS.application_name = "geomint"
SETTINGS.application_version = "0.0.1"
SETTINGS.dataspace = config.RDDMSDATASPACE
SETTINGS.data_partition = config.OSDUPARTITION
pyetp_logger.setLevel(logging.WARNING)

