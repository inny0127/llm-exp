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
        self.py(f"""
import subprocess, os
if not os.path.exists('/content/llm-exp'):
    subprocess.run('git clone -q -b {self.a.branch} {REPO} /content/llm-exp', shell=True, check=True)
else:
    subprocess.run('cd /content/llm-exp && git pull -q origin {self.a.branch}', shell=True, check=True)
os.makedirs('{REMOTE}/data', exist_ok=True); os.makedirs('{REMOTE}/runs', exist_ok=True)
print(subprocess.run('cd /content/llm-exp && git log --oneline -1', shell=True, capture_output=True, text=True).stdout)
""", timeout=300)
        # restore the mirror (no logs, no optimizer state once pretraining is done)
        done = os.path.exists(os.path.join(self.mirror, "runs/pretrain/done"))
        n = 0
        for root, _, files in os.walk(self.mirror):
            for f in files:
                p = os.path.join(root, f)
                rel = os.path.relpath(p, self.mirror)
                if rel.startswith(".") or rel == "orchestrator.log" or "/logs/" in rel or rel.endswith(".log"):
                    continue
                if done and rel.endswith("resume.pt"):
                    continue
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
        return self.py("import subprocess\nprint(subprocess.run('pgrep -f [r]un_exp03.py', shell=True,"
                       " capture_output=True, text=True).stdout.strip() != '')") == "True"

    def start_driver(self):
        cmd = (f"cd /content/llm-exp/exp03_scale && nohup python -u run_exp03.py --stage all --data {REMOTE}/data"
               f" --runs {REMOTE}/runs --device cuda --workers {self.a.workers} --budget_cu 1e9"
               f" --cu_per_hour {self.a.cu_per_hour} >> {REMOTE}/driver.log 2>&1 &")
        self.py(f"import subprocess\nsubprocess.Popen({cmd!r}, shell=True)\nprint('started')")
        self.log("driver started")

    # ------------------------------------------------------------ mirror
    def manifest(self):
        out = self.py(f"""
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
""", timeout=120)
        return json.loads(out)

    def want(self, rel, size, man):
        if rel.endswith("model.pt") and "runs/pretrain/done" not in man:
            return False  # written non-atomically mid-pretraining; resume.pt holds the same weights
        if rel.endswith("resume.pt"):
            if "runs/pretrain/done" in man or time.time() - self.last_resume_dl < 40 * 60:
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
            dst = os.path.join(self.mirror, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            rc, out = self.run(["download", "-s", self.a.session, f"{REMOTE}/{rel}", dst + ".tmp"], timeout=3600)
            if rc:
                self.log("download failed:", rel, out[-200:])
                continue
            os.replace(dst + ".tmp", dst)
            self.state[rel] = [size, mtime]
            json.dump(self.state, open(self.state_path, "w"))
            got += 1
            if rel.endswith("resume.pt"):
                self.last_resume_dl = time.time()
        return got, man

    def tail(self, n=1200):
        return self.py(f"import os\np='{REMOTE}/driver.log'\nprint(open(p).read()[-{n}:] if os.path.exists(p) else '')")

    # ------------------------------------------------------------ main loop
    def loop(self):
        fresh = True
        while True:
            try:
                if not self.alive():
                    self.log("no runtime; creating one")
                    self.new_session()
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
