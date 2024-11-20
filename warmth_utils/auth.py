import msal
from warmth_utils.config import config

if config.AUTH_MODE == "Azure":
    msal_client = msal.ConfidentialClientApplication(str(config.CLIENT_ID), config.CLIENT_SECRET,f'https://login.microsoftonline.com/{config.TENANT_ID}')
else:
    msal_client = None

def msal_token():
    if isinstance(msal_client, type(None)):
        return None
    else:
        resp = msal_client.acquire_token_for_client([f'{config.OSDURESOURCEID}/.default'])
        return f"Bearer {resp['access_token']}"
        