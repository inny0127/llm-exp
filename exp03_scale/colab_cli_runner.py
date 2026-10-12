"""Drive run_exp03.py on a Colab runtime through the Colab CLI, from another machine.

Keeps a local mirror of everything the run needs to resume (data, checkpoints, finished runs),
re-creates the runtime when Colab reclaims it, restores the mirror onto it and restarts the
driver, whose stages skip finished work. Stops the runtime when the driver has packaged results.

Usage: python colab_cli_runner.py --mirror ~/exp03_mirror [--gpu T4] [--branch BRANCH]
Needs `colab` (pip install google-colab-cli) already authenticated.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

REMOTE = "/content/exp03"
REPO = "https://github.com/inny0127/llm-exp.git"
BIG = 50 * 2**20


class Runner:
    def __init__(self, a):
        self.a = a
        self.cli = [a.colab, "--auth", "oauth2"]
        self.mirror = os.path.abspath(os.path.expanduser(a.mirror))
        os.makedirs(self.mirror, exist_ok=True)
        self.state_path = os.path.join(self.mirror, ".state.json")
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {}
        self.last_resume_dl = 0.0
        self.crashes = 0
        self.api_broken = False  # set when this runtime's file API returns 500

    def log(self, *msg):
        line = time.strftime("%H:%M:%S ") + " ".join(map(str, msg))
        print(line, flush=True)
        with open(os.path.join(self.mirror, "orchestrator.log"), "a") as f:
            f.write(line + "\n")

    def run(self, args, stdin=None, timeout=600):
        try:
            r = subprocess.run(self.cli + args, input=stdin, capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return -1, "timeout"

    def py(self, code, timeout=90):
        """Run Python on the runtime; returns stdout between markers (robust to CLI chatter)."""
        wrapped = f"print('<<<')\n{code}\nprint('>>>')"
        rc, out = self.run(["exec", "-s", self.a.session, "--timeout", str(timeout)], wrapped, timeout + 60)
        if "<<<" in out and ">>>" in out:
            return out.split("<<<", 1)[1].rsplit(">>>", 1)[0].strip()
        raise RuntimeError(out[-800:])

    def scp(self, src, dst, timeout=3600):
        """scp through `colab ssh --proxy-mode` (about 20 MB/s); remote paths start with ':'."""
        if not shutil.which("scp"):
            return False
        proxy = f"ProxyCommand={' '.join(self.cli)} ssh --proxy-mode -s {self.a.session}"
        fix = lambda x: "root@colab" + x if x.startswith(":") else x  # noqa: E731
        try:
            r = subprocess.run(["scp", "-q", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                                "-o", "LogLevel=ERROR", "-o", proxy, fix(src), fix(dst)],
                               capture_output=True, text=True, timeout=timeout)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def ssh(self, script, timeout=600):
        """Run a bash script on the runtime over SSH (steadier than the kernel channel)."""
        proxy = f"ProxyCommand={' '.join(self.cli)} ssh --proxy-mode -s {self.a.session}"
        r = subprocess.run(["ssh", "-q", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                            "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=30", "-o", proxy, "root@colab",
                            "bash -s"], input=script, capture_output=True, text=True, timeout=timeout)
        if r.returncode:
            raise RuntimeError(f"ssh rc {r.returncode}: {(r.stdout + r.stderr)[-500:]}")
        return r.stdout

    def alive(self):
        rc, out = self.run(["sessions"], timeout=120)
        return f"[{self.a.session}]" in out

    # ------------------------------------------------------------ session setup
    def new_session(self):
        while True:
            rc, out = self.run(["new", "-s", self.a.session, "--gpu", self.a.gpu], timeout=600)
            if "READY" in out:
                self.log("runtime ready:", self.a.gpu)
                return
            self.log("could not get a runtime; retrying in 20 min:", out.strip()[-300:])
            time.sleep(1200)

    def setup(self):
        self.log(self.ssh(f"""set -e
if [ -d /content/llm-exp ]; then cd /content/llm-exp && git pull -q origin {self.a.branch}
else git clone -q -b {self.a.branch} {REPO} /content/llm-exp; fi
mkdir -p {REMOTE}/data {REMOTE}/runs
cd /content/llm-exp && git log --oneline -1
""", timeout=300).strip())
        # restore the mirror (no logs, no optimizer state once pretraining is done)
        done = os.path.exists(os.path.join(self.mirror, "runs/pretrain/done"))
        n = 0
        for root, _, files in os.walk(self.mirror):
            for f in files:
                p = os.path.join(root, f)
                rel = os.path.relpath(p, self.mirror)
                if rel.startswith(".") or rel in ("orchestrator.log", "runner.out") or "/logs/" in rel or rel.endswith(".log"):
                    continue
                if done and rel.endswith("resume.pt"):
                    continue
                if "/state/" in rel and os.path.exists(os.path.join(self.mirror, rel.replace("/state/", "/")[:-3] + ".json")):
                    continue  # that run finished; its resume state is stale
                self.ssh(f"mkdir -p '{os.path.dirname(f'{REMOTE}/{rel}')}'", timeout=120)
                if not self.scp(p, f":{REMOTE}/{rel}"):
                    rc, out = self.run(["upload", "-s", self.a.session, p, f"{REMOTE}/{rel}"], timeout=3600)
                    if rc:
                        raise RuntimeError(f"upload failed for {rel}: {out[-300:]}")
                n += 1
        if n:  # the uploaded copies match the mirror: do not download them back
            for rel, sm in self.manifest().items():
                p = os.path.join(self.mirror, rel)
                if os.path.exists(p) and os.path.getsize(p) == sm[0]:
                    self.state[rel] = sm
            json.dump(self.state, open(self.state_path, "w"))
        self.log(f"restored {n} files")

    def driver_running(self):
        return self.ssh("pgrep -f '[r]un_exp03.py' >/dev/null && echo True || echo False", 120).strip() == "True"

    def start_driver(self):
        self.ssh(f"cd /content/llm-exp/exp03_scale && setsid nohup python3 -u run_exp03.py --stage all"
                 f" --data {REMOTE}/data --runs {REMOTE}/runs --device cuda --workers {self.a.workers}"
                 f" --budget_cu 1e9 --cu_per_hour {self.a.cu_per_hour} >> {REMOTE}/driver.log 2>&1 < /dev/null &",
                 120)
        self.log("driver started")

    # ------------------------------------------------------------ mirror
    def manifest(self):
        out = self.ssh(f"""python3 - <<'PY'
import os, json
m = {{}}
for top in ('data', 'runs'):
    for r, ds, fs in os.walk('{REMOTE}/' + top):
        ds[:] = [d for d in ds if d not in ('raw', 'smoke')]
        for f in fs:
            if f.endswith(('.tmp', '.part')):
                continue
            p = os.path.join(r, f); st = os.stat(p)
            m[os.path.relpath(p, '{REMOTE}')] = [st.st_size, int(st.st_mtime)]
print(json.dumps(m))
PY
""", timeout=180)
        return json.loads(out.strip().splitlines()[-1])

    def want(self, rel, size, man):
        if rel.endswith("model.pt") and "runs/pretrain/done" not in man:
            return False  # written non-atomically mid-pretraining; resume.pt holds the same weights
        if rel.endswith("resume.pt"):
            if "runs/pretrain/done" in man or time.time() - self.last_resume_dl < 20 * 60:
                return False
        if rel.startswith("data/") and size > BIG and "runs/smoke_ok.json" not in man:
            return False  # data files are final once the smoke stage has run
        return True

    def sync(self):
        man = self.manifest()
        got = 0
        for rel, (size, mtime) in sorted(man.items(), key=lambda kv: kv[1][0]):
            if self.state.get(rel) == [size, mtime] or not self.want(rel, size, man):
                continue
            if "/logs/" in rel and rel.replace("/logs/", "/").rsplit(".", 1)[0] + ".json" not in man:
                continue
            dst = os.path.join(self.mirror, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            ok = self.scp(f":{REMOTE}/{rel}", dst + ".dl")
            if ok and os.path.getsize(dst + ".dl") != size:
                continue  # changed while copying: take it next round
            if not ok and not self.api_broken:
                rc, out = self.run(["download", "-s", self.a.session, f"{REMOTE}/{rel}", dst + ".dl"], timeout=3600)
                ok = rc == 0 and "failed" not in out.lower()
                self.api_broken = not ok and "500" in out
            if ok:
                os.replace(dst + ".dl", dst)
            else:
                try:  # the runtime's file API sometimes returns 500: pull it through the kernel instead
                    self.fetch(rel, size, dst + ".tmp")
                except Exception as e:  # noqa: BLE001
                    self.log("download failed:", rel, str(e)[-200:])
                    continue
                os.replace(dst + ".tmp", dst)
            self.state[rel] = [size, mtime]
            json.dump(self.state, open(self.state_path, "w"))
            got += 1
            if rel.endswith("resume.pt"):
                self.last_resume_dl = time.time()
        return got, man

    def fetch(self, rel, size, dst, chunk=4 * 2**20):
        """Copy a remote file in base64 chunks over `colab exec`, resuming a partial copy;
        checked by size and sha256."""
        import base64
        import hashlib
        have = os.path.getsize(dst) if os.path.exists(dst) else 0
        have -= have % chunk
        with open(dst, "r+b" if os.path.exists(dst) else "wb") as f:
            f.truncate(have)
            f.seek(have)
            for off in range(have, max(size, 1), chunk):
                b = self.py(f"import base64\nwith open('{REMOTE}/{rel}','rb') as f:\n    f.seek({off})\n"
                            f"    print(base64.b64encode(f.read({chunk})).decode())", timeout=300)
                f.write(base64.b64decode(b))
                f.flush()
        h = hashlib.sha256()
        with open(dst, "rb") as f:
            for c in iter(lambda: f.read(1 << 24), b""):
                h.update(c)
        remote = self.py(f"import hashlib\nh=hashlib.sha256()\nwith open('{REMOTE}/{rel}','rb') as f:\n"
                         f"    [h.update(c) for c in iter(lambda: f.read(1<<24), b'')]\nprint(h.hexdigest())",
                         timeout=300)
        if os.path.getsize(dst) != size or remote != h.hexdigest():
            os.remove(dst)
            raise RuntimeError(f"{rel}: size or checksum mismatch (changed while copying?)")

    def tail(self, n=1200):
        return self.ssh(f"tail -c {n} {REMOTE}/driver.log 2>/dev/null || true", 120)

    # ------------------------------------------------------------ main loop
    def loop(self):
        fresh = True
        while True:
            try:
                if not self.alive():
                    self.log("no runtime; creating one")
                    self.new_session()
                    self.api_broken = False
                    self.setup()
                    fresh = True
                elif self.ssh(f"test -d /content/llm-exp && test -f {REMOTE}/data/train.bin && echo ok || echo no",
                              120).strip() != "ok":
                    self.log("runtime up but not set up; restoring")
                    self.setup()
                    fresh = True
                if not self.driver_running():
                    t = self.tail(4000)
                    if "packaged:" in t and not fresh:
                        self.sync()
                        self.log("driver finished; results mirrored. Stopping the runtime.")
                        self.run(["stop", "-s", self.a.session], timeout=300)
                        return 0
                    if not fresh:
                        self.crashes += 1
                        self.log(f"driver stopped without packaging (crash {self.crashes}):\n" + t[-2000:])
                        if self.crashes >= 3:
                            self.log("giving up after 3 crashes; runtime left up for inspection")
                            return 1
                    self.start_driver()
                    fresh = False
                got, man = self.sync()
                stage = [l for l in self.tail(3000).splitlines() if l.startswith("=====")]
                self.log(f"alive; stage {stage[-1] if stage else '?'}; mirrored {got} new files;"
                         f" {sum(1 for k in man if k.startswith('runs/test/') and k.endswith('.json'))} test runs done")
            except Exception as e:  # noqa: BLE001  (CLI hiccups: retry next round)
                self.log("round failed:", str(e)[-400:])
            time.sleep(self.a.every)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--colab", default="colab")
    ap.add_argument("--session", default="exp03")
    ap.add_argument("--gpu", default="T4")
    ap.add_argument("--branch", default="claude/experiment-design-run-sjkj74")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cu_per_hour", type=float, default=0.0, help="0 on the free tier")
    ap.add_argument("--every", type=int, default=300, help="seconds between rounds")
    sys.exit(Runner(ap.parse_args()).loop())


if __name__ == "__main__":
    main()
