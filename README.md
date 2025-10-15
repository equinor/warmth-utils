# Install
`pip install git+https://github.com/equinor/warmth-utils.git`

# Environmental variables required
- CLIENT_ID
- CLIENT_SECRET
- TENANT_ID
- CACHEDIR
- MODELSPEC
- RDDMSURL
- RDDMSDATASPACE
- OSDUHOST
- OSDUPARTITION
- OSDURESOURCEID

# Get model spec
```python
from warmth_utils.geomint import model_spec_dict, model_spec
model_spec_dict # Model spec in python dict
model_spec # Optional: Model spec in Pydantic object
```
# Download map and save as irap_binary file
```python
from warmth_utils.rddms import download_map
from warmth_utils.config import CACHE_DIR
import asyncio
epc = "eml:///dataspace('psscloud/Dev')/eml20.EpcExternalPartReference(004fa1b4-cf6e-4b35-a933-2f98d1430c8d)"
crs = "eml:///dataspace('psscloud/Dev')/resqml20.LocalDepth3dCrs(96605b7a-5f14-4de8-9548-c59887d8964c)"
gri = "eml:///dataspace('psscloud/Dev')/resqml20.Grid2dRepresentation(ccc012b4-d095-4c39-9c9a-18f8ecfbc92e)"
filepath = CACHE_DIR / <FILENAME>
asyncio.run(download_map(epc, gri, str(filepath)))
```

# Download epc mesh
```python
from warmth_utils.rddms import download_epc
import asyncio
mesh_path = asyncio.run(download_epc())
mesh_path = str(mesh_path) # as string
```

# Upload Migris result maps
```python
from warmth_utils.rddms import connect
from warmth_utils.auth import msal_token
import xtgeo
import asyncio
async def upload_maps(surf):
    async with connect(msal_token()) as client:
        uris = await client.put_xtgeo_surface(surf, 23031)
    return list(map(lambda u: u.raw_uri, uris))
surf = xtgeo.surface_from_file(<FILEPATH>, fformat="irap_binary")
r = asyncio.run(upload_maps(surf))
```

# Upload Migris results
```python
from warmth_utils.osdu import upload_migri_results
from warmth_utils.geomint import model_spec
import json
# metadata_json_path is the json frile from post_migribee step
result_maps_metadata_path = <metadata_json_path>
with open(result_maps_metadata_path, "r") as f:
    result_maps_metadata = json.load(f)
upload_migri_results(<PATH_TO_MIGRI_PROJECT_FILE>, result_maps_metadata)
```

# AUTH MODE TO RDDMS
```python
# No auth
from warmth_utils.config import config
config.AUTH_MODE = "None"

# Azure (default)
from warmth_utils.config import config
config.AUTH_MODE = "Azure"
```

###
poetry run datamodel-codegen  --input ./warmth_utils/swagger.json --input-file-type openapi --output ./warmth_utils/datamodels

# Testing
First, start the open-etp-server via:
```bash
docker compose -f tests/compose.yml up
```
Then, start a new terminal and run:
```bash
poetry run py.test
```
Note that the listed environment variables (described higher up) have to be set
to some values that gets validated correctly. See `UTIL_SETTINGS` in
[`warmth_utils/config.py`](warmth_utils/config.py) for types.
