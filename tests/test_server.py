"""HTTP contract and invalid-input recovery for the standalone service."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from armiger.server import create_server


@pytest.fixture
def service():
    class Pipeline:
        def classify(self, item):
            return {"category": "movie", "name": item["name"]}

    server = create_server(Pipeline(), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def test_classification_and_health(service):
    health = json.load(urllib.request.urlopen(service + "/healthz"))
    assert health["status"] == "ok"
    for key in ("active_seq", "active_bundle_merkle_root", "model_source", "last_dht_check_ts", "sharing_status"):
        assert key in health
    request = urllib.request.Request(service + "/v1/classify", data=b'{"name":"Sintel.2010.mkv"}', headers={"Content-Type": "application/json"})
    assert json.load(urllib.request.urlopen(request))["category"] == "movie"


@pytest.mark.parametrize("body", [b"not json", b"[]", b"{}", b'{"name":3}'])
def test_bad_request_does_not_break_service(service, body):
    request = urllib.request.Request(service + "/v1/classify", data=body)
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)
    assert error.value.code == 400
    assert json.load(urllib.request.urlopen(service + "/healthz"))["status"] == "ok"
