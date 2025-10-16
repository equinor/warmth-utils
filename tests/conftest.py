import uuid

import numpy as np
import pytest
import pytest_asyncio

from pyetp.client import ETPClient, connect, ETPError
from warmth_utils.config import UTIL_SETTINGS, SETTINGS


SETTINGS.application_name = "warmth-utils-testing"
SETTINGS.etp_url = "ws://localhost:9100"
SETTINGS.etp_timeout = 30
dataspace = "test/test"


@pytest_asyncio.fixture
async def etp_client():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ws_open = sock.connect_ex(("127.0.0.1", 9100)) == 0

    if not ws_open:
        pytest.skip(
            reason="websocket for test server not open", allow_module_level=True
        )

    async with connect() as client:
        yield client


@pytest_asyncio.fixture
async def duri(etp_client: ETPClient):
    uri = etp_client.dataspace_uri("test/test")
    try:
        resp = await etp_client.put_dataspaces(uri)
        yield uri
    except ETPError as e:
        # We typically get an error if the dataspace already exists (code=5).
        # In this case, ignore the error.
        yield uri
    finally:
        resp = await etp_client.delete_dataspaces(uri)
        assert len(resp) == 1, "should cleanup our test dataspace"
