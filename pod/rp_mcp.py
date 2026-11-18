"""Minimal client for the RunPod MCP server (https://mcp.runpod.io/), OAuth token from a local file.

Usage: python rp_mcp.py TOKEN_FILE tools | call NAME 'JSON_ARGS'
"""
import json
import sys
import urllib.request

URL = "https://mcp.runpod.io/"


def rpc(tok, method, params, sid=None, rid=1):
    h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
         "Authorization": "Bearer " + tok, "MCP-Protocol-Version": "2025-06-18"}
    if sid:
        h["Mcp-Session-Id"] = sid
    body = {"jsonrpc": "2.0", "method": method, "params": params}
    if rid is not None:
        body["id"] = rid
    r = urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(), headers=h), timeout=120)
    sid = r.headers.get("Mcp-Session-Id", sid)
    raw = r.read().decode()
    if not raw.strip():
        return None, sid
    if raw.lstrip().startswith("{"):
        return json.loads(raw), sid
    msgs = [json.loads(l[5:]) for l in raw.splitlines() if l.startswith("data:")]
    return (msgs[-1] if msgs else None), sid


def session(tok):
    _, sid = rpc(tok, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                     "clientInfo": {"name": "llm-exp", "version": "1"}})
    rpc(tok, "notifications/initialized", {}, sid, rid=None)
    return sid


def main():
    tok = json.load(open(sys.argv[1]))["access_token"]
    sid = session(tok)
    if sys.argv[2] == "tools":
        out, _ = rpc(tok, "tools/list", {}, sid, 2)
        for t in out["result"]["tools"]:
            print(t["name"], "-", t.get("description", "")[:110])
    else:
        out, _ = rpc(tok, "tools/call", {"name": sys.argv[3], "arguments": json.loads(sys.argv[4] if len(sys.argv) > 4 else "{}")}, sid, 2)
        res = out.get("result", out)
        for c in res.get("content", []):
            print(c.get("text", c))
        if "content" not in res:
            print(json.dumps(res)[:3000])


if __name__ == "__main__":
    main()
