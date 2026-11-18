"""Fetch files from the pod in base64 chunks over the RunPod MCP exec-pod tool.

Usage: python pfetch.py POD_ID TOKEN_FILE REMOTE_DIR 'GLOB ...' LOCAL_DIR
"""
import base64
import io
import json
import subprocess
import sys
import tarfile

pod, tok, rdir, globs, ldir = sys.argv[1:6]


def ex(cmd):
    arg = json.dumps({"id": pod, "body": {"command": ["sh", "-c", cmd]}})
    out = subprocess.run([sys.executable, "rp_mcp.py", tok, "call", "exec-pod", arg], capture_output=True, text=True).stdout
    return json.loads(out)["stdout"]


n = int(ex(f"cd {rdir} && tar czf /workspace/_f.tgz {globs} && base64 -w0 /workspace/_f.tgz > /workspace/_f.b64 && wc -c < /workspace/_f.b64").strip())
b64 = "".join(ex(f"dd if=/workspace/_f.b64 bs=1 skip={o} count=30000 2>/dev/null") for o in range(0, n, 30000))
tarfile.open(fileobj=io.BytesIO(base64.b64decode(b64))).extractall(ldir)
print("fetched", n, "b64 bytes into", ldir)
