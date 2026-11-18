"""Create, inspect, fetch from and terminate the RunPod pod that runs pod_start.sh.

The API key is injected by the session's network proxy for api.runpod.io; it never appears here.
Usage: python runpod_ctl.py create | status ID | fetch ID PATH DEST | terminate ID
"""
import json
import sys
import urllib.request

API = "https://api.runpod.io/graphql"
START = ("bash -c 'cd /workspace && (git clone -q -b claude/experiment-design-run-sjkj74 "
         "https://github.com/inny0127/llm-exp.git || true) && bash llm-exp/pod/pod_start.sh'")


def gql(q):
    req = urllib.request.Request(API, data=json.dumps({"query": q}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "llm-exp"})
    out = json.load(urllib.request.urlopen(req, timeout=60))
    if out.get("errors"):
        raise SystemExit(json.dumps(out["errors"], indent=1))
    return out["data"]


def main():
    cmd = sys.argv[1]
    if cmd == "create":
        gpu = sys.argv[2] if len(sys.argv) > 2 else "NVIDIA GeForce RTX 4090"
        cloud = sys.argv[3] if len(sys.argv) > 3 else "COMMUNITY"
        d = gql(f'''mutation {{ podFindAndDeployOnDemand(input: {{ cloudType: {cloud}, gpuCount: 1,
          gpuTypeId: "{gpu}", name: "llm-exp", imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
          containerDiskInGb: 30, volumeInGb: 0, minVcpuCount: 8, minMemoryInGb: 24, ports: "8000/http",
          dockerArgs: {json.dumps(START)} }}) {{ id machineId costPerHr }} }}''')
        print(json.dumps(d["podFindAndDeployOnDemand"]))
    elif cmd == "status":
        d = gql(f'query {{ pod(input: {{podId: "{sys.argv[2]}"}}) {{ id desiredStatus costPerHr '
                f'runtime {{ uptimeInSeconds gpus {{ gpuUtilPercent }} }} }} }}')
        print(json.dumps(d["pod"]))
    elif cmd == "fetch":
        url = f"https://{sys.argv[2]}-8000.proxy.runpod.net/{sys.argv[3]}"
        data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "llm-exp"}), timeout=120).read()
        open(sys.argv[4], "wb").write(data)
        print(len(data), "bytes")
    elif cmd == "terminate":
        print(gql(f'mutation {{ podTerminate(input: {{podId: "{sys.argv[2]}"}}) }}'))


if __name__ == "__main__":
    main()
