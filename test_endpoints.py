import requests
import json

url = "http://127.0.0.1:7860"

def test_endpoint(name, method, path, **kwargs):
    try:
        if method == "GET":
            r = requests.get(url + path, **kwargs)
        else:
            r = requests.post(url + path, **kwargs)
        print(f"{name}: {r.status_code}")
        if r.status_code != 200:
            print(f"  Error: {r.text}")
    except Exception as e:
        print(f"{name}: FAILED - {e}")

test_endpoint("Tasks", "GET", "/tasks")
test_endpoint("Health", "GET", "/health")
test_endpoint("Metadata", "GET", "/metadata")
test_endpoint("Reset", "POST", "/reset", json={})
test_endpoint("State", "GET", "/state")
test_endpoint("Step", "POST", "/step", json={"action_type": "do_nothing", "internal_thought": "testing step"})
