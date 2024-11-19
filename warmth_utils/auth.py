import msal
from warmth_utils.config import config


msal_client = msal.ConfidentialClientApplication(str(config.CLIENT_ID), config.CLIENT_SECRET,f'https://login.microsoftonline.com/{config.TENANT_ID}')

def msal_token():
    if config.AUTH_MODE == "Azure":
        resp = msal_client.acquire_token_for_client([f'{config.OSDURESOURCEID}/.default'])
        return f"Bearer {resp['access_token']}"
    else:
        return None