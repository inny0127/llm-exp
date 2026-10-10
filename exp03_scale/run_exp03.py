"""Exp 03 driver: the frozen Exp 02 recipe at ~30M dense parameters and 2,000-fact streams, on one GPU.

Every stage is idempotent and skips work already on disk. With --mirror (a Google Drive folder),
data, checkpoints and finished runs are copied there as they appear and restored on start, so a
Colab session that disconnects resumes by rerunning the same cell.

Spend is tracked as runtime-connected time of this script per VM boot, converted to compute units
with --cu_per_hour; no new run starts once --budget_cu would be exceeded.

Usage (Colab):  python run_exp03.py --stage all --mirror /content/drive/MyDrive/llm-exp-exp03
Local dry run:  python run_exp03.py --stage all --dry --device cpu --data /tmp/d --runs /tmp/r
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "..", "exp01_two_speed_weights")
EXP02 = os.path.join(HERE, "..", "exp02_fix_bottlenecks")
HF = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
TRAIN, VALID = "TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt"

P06 = ["--write_loss", "answer", "--protect_gamma", "1", "--protect_templates", "stream"]
GENTLE32 = ["--sleep", "--sleep_fact_loss", "answer", "--sleep_rehearse", "32", "--rehearse_labels",
            "teacher", "--rehearse_templates", "nonprobe", "--sleep_freeze", "embed", "--clean_replay"]
ARMS = {  # name -> stream.py flags (LR and seed added per run)
    "C": ["--arm", "C"],
    "p06ans": ["--arm", "C"] + P06,
    "C+gentle32": ["--arm", "C"] + GENTLE32,
    "p06ans+gentle32": ["--arm", "C"] + P06 + GENTLE32,
    "A": ["--arm", "A"],
}
# Dev grids (seed 0). The scale change can move every LR, so each writer gets a grid around its
# Exp 02 value; DESIGN.md gives the selection rules.
DEV = {"C": [16, 32, 64], "p06ans": [2, 4, 8, 16], "A": [3e-6, 1e-5, 3e-5]}
TEST_SEEDS = [1, 2, 3]
SLEEP_EVERY, P_RECENT, P_OLD = 100, 7, 2  # stream.py defaults; used to project sleep cost
STAGES = ["data", "smoke", "pretrain", "prep", "probe", "dev", "freeze", "test", "analyze", "package"]


def sh(cmd, cwd=ENGINE, log=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    with open(log, "w") if log else open(os.devnull, "w") as f:
        r = subprocess.run(cmd, cwd=cwd, stdout=f if log else None, stderr=subprocess.STDOUT if log else None)
    if r.returncode:
        if log:
            print(open(log).read()[-3000:])
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}")


# ---------------------------------------------------------------- Drive mirror
_sync_lock = threading.Lock()


def sync(src, dst, skip=("raw", "smoke")):
    """Copy files that are missing or changed (size or mtime) from src to dst; atomic per file."""
    if not dst or not os.path.isdir(src):
        return
    with _sync_lock:
        for root, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if d not in skip]
            rel = os.path.relpath(root, src)
            for f in files:
                if f.endswith((".tmp", ".part")):
                    continue
                s, d = os.path.join(root, f), os.path.join(dst, rel, f)
                st = os.stat(s)
                if os.path.exists(d):
                    dt = os.stat(d)
                    if dt.st_size == st.st_size and int(dt.st_mtime) >= int(st.st_mtime):
                        continue
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copyfile(s, d + ".tmp")
                try:  # keep mtimes so unchanged files are skipped next time
                    os.utime(d + ".tmp", (st.st_atime, st.st_mtime))
                except OSError:
                    pass
                os.replace(d + ".tmp", d)


def mirror(a):
    if a.mirror:
        sync(a.data, os.path.join(a.mirror, "data"))
        sync(a.runs, os.path.join(a.mirror, "runs"))


# ---------------------------------------------------------------- budget
class Budget:
    """Script wall time per VM boot (a Colab runtime), turned into compute units."""

    def __init__(self, a):
        self.path, self.a = os.path.join(a.runs, "budget.json"), a
        self.d = json.load(open(self.path)) if os.path.exists(self.path) else {"stages": {}, "sessions": {}}
        try:
            self.boot = open("/proc/sys/kernel/random/boot_id").read().strip()
        except OSError:
            self.boot = "local"
        s = self.d["sessions"].setdefault(self.boot, {"seconds": 0.0})
        self.t0, self.base, self.lock = time.time(), s["seconds"], threading.Lock()
        threading.Thread(target=self._heartbeat, daemon=True).start()

    def _heartbeat(self):  # a VM that dies mid-stage still leaves its spend on disk
        while True:
            time.sleep(120)
            self.save()

    def tick(self):
        self.d["sessions"][self.boot]["seconds"] = self.base + time.time() - self.t0

    def add(self, stage, seconds):
        self.d["stages"][stage] = self.d["stages"].get(stage, 0) + seconds
        self.save()

    def save(self):
        with self.lock:
            self.tick()
            json.dump(self.d, open(self.path + ".tmp", "w"), indent=1)
            os.replace(self.path + ".tmp", self.path)

    def spent_cu(self):
        self.tick()
        return sum(s["seconds"] for s in self.d["sessions"].values()) / 3600 * self.a.cu_per_hour

    def left_cu(self):
        return self.a.budget_cu - self.spent_cu()

    def report(self):
        for k, v in self.d["stages"].items():
            print(f"  {k:9s} {v / 60:7.1f} min", flush=True)
        print(f"  script time over {len(self.d['sessions'])} session(s) ~{self.spent_cu():.1f} CU of"
              f" --budget_cu {self.a.budget_cu} at {self.a.cu_per_hour} CU/h (Colab's own meter is authoritative)",
              flush=True)


# ---------------------------------------------------------------- runs
def stream_cmd(a, arm, lr, seed, out, n_facts):
    name = f"{arm}_lr{lr:g}_s{seed}"
    cmd = [sys.executable, "stream.py", *ARMS[arm], "--lr", f"{lr:g}", "--seed", str(seed),
           "--n_facts", str(n_facts), "--ckpt_every", str(max(1, n_facts // 8)), "--out", out, "--name", name,
           "--device", a.device, "--data_dir", a.data, "--pretrained", os.path.join(a.runs, "pretrain", "model.pt"),
           "--bg_windows", str(a.bg_windows)]
    if a.dry:  # a few sleeps inside the 40-fact dry streams
        cmd += ["--sleep_every", "10"]
    return name, cmd


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # inherited by every run


def run_pool(a, budget, jobs, out, workers=None):
    """Run stream jobs in parallel; finished runs are kept, and no job starts past the budget."""
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    todo = [(n, c) for n, c in jobs if not os.path.exists(os.path.join(out, n + ".json"))]
    workers = workers or a.workers
    print(f"{len(jobs) - len(todo)} done, {len(todo)} to run with {workers} workers", flush=True)

    def one(job):
        n, c = job
        if budget.left_cu() <= 0 and not a.dry:
            print(f"skipped {n}: budget reached", flush=True)
            return "skipped"
        t0 = time.time()
        with open(os.path.join(out, "logs", n + ".log"), "w") as f:
            rc = subprocess.run(c, cwd=ENGINE, stdout=f, stderr=subprocess.STDOUT).returncode
        print(f"{'done' if rc == 0 else f'FAILED({rc})'} {n} ({(time.time() - t0) / 60:.1f} min,"
              f" ~{budget.spent_cu():.1f} CU so far)", flush=True)
        budget.save()
        mirror(a)
        return rc

    with ThreadPoolExecutor(workers) as ex:
        rcs = list(ex.map(one, todo))
    if "skipped" in rcs:
        raise SystemExit("budget reached; raise --budget_cu and rerun this stage (finished runs are kept)")
    if any(rcs):
        raise SystemExit("some runs failed; see logs, then rerun this stage (finished runs are kept)")


def load(path):
    return json.load(open(path))


# ---------------------------------------------------------------- stages
def st_data(a, budget):
    need = ["train.bin", "val.bin", "facts.json", "tokenizer.json"]
    if all(os.path.exists(os.path.join(a.data, f)) for f in need):
        return print("data: present")
    raw = os.path.join(a.data, "raw")
    os.makedirs(raw, exist_ok=True)
    files = [VALID] if a.dry else [TRAIN, VALID]  # the dry run trains on the valid split
    for f in files:
        p = os.path.join(raw, f)
        if not os.path.exists(p):
            print("downloading", f, flush=True)
            urllib.request.urlretrieve(f"{HF}/{f}", p + ".part")
            os.replace(p + ".part", p)
    # Exp 01's tokenizer: names split into the same syllable pieces at both scales
    shutil.copy(os.path.join(HERE, "tokenizer.json"), os.path.join(a.data, "tokenizer.json"))
    sh([sys.executable, "data.py", "--data_dir", a.data, "--train_file", VALID if a.dry else TRAIN,
        "--max_mb", "0", "--pretrain_tokens", str(a.pretrain_tokens), "--n_old", str(a.n_old),
        "--bios_per_old", str(a.bios_per_old), "--n_streams", str(1 + len(TEST_SEEDS)),
        "--stream_len", str(a.n_facts)], log=os.path.join(a.runs, "data.log"))
    print(open(os.path.join(a.runs, "data.log")).read()[-1500:], flush=True)
    shutil.rmtree(raw)  # 2.2 GB; train.bin is all the runs need


def st_smoke(a, budget):
    """Real-batch pretraining steps (times them) and a 12-fact stream of the two most complex arms."""
    d = os.path.join(a.runs, "smoke")
    ok = os.path.join(a.runs, "smoke_ok.json")
    if os.path.exists(ok):
        print("smoke: passed earlier", open(ok).read())
        return check_pretrain_cost(a, budget, load(ok))
    shutil.rmtree(d, ignore_errors=True)
    n = 20 if a.dry else 60
    sh([sys.executable, "pretrain.py", "--size", a.size, "--device", a.device, "--data_dir", a.data,
        "--out", os.path.join(d, "pretrain"), "--max_steps", str(n), "--eval_every", str(n),
        "--batch", str(a.batch)] + (["--amp"] if a.device.startswith("cuda") else []),
       log=os.path.join(a.runs, "smoke_pretrain.log"))
    rec = [json.loads(l) for l in open(os.path.join(d, "pretrain", "log.jsonl"))][-1]
    pre = os.path.join(d, "pretrain", "model.pt")
    for arm in ("p06ans+gentle32", "A"):
        sh([sys.executable, "stream.py", *ARMS[arm], "--lr", "4" if arm != "A" else "1e-5", "--seed", "0",
            "--n_facts", "12", "--ckpt_every", "12", "--sleep_every", "6", "--steps", "3", "--out", d,
            "--name", f"smoke_{arm}", "--device", a.device, "--data_dir", a.data, "--pretrained", pre,
            "--bg_windows", "2"], log=os.path.join(a.runs, f"smoke_{arm}.log"))
    # (the smoke runs' bg_df / protect caches belong to the smoke model and stay in smoke/pretrain;
    # the eval-document mask depends only on the data and is reused by every later run)
    sec_per_step = rec["elapsed_s"] / n  # includes CUDA start-up: pessimistic
    ctx = 128 if a.size == "tiny" else 256
    steps = a.pretrain_tokens // ((ctx + 1) * a.batch)
    proj = {"sec_per_step": sec_per_step, "pretrain_steps": steps,
            "pretrain_minutes": sec_per_step * steps / 60,
            "pretrain_cu": sec_per_step * steps / 3600 * a.cu_per_hour}
    print("smoke: passed", json.dumps(proj), flush=True)
    check_pretrain_cost(a, budget, proj)
    json.dump(proj, open(ok, "w"), indent=1)


def check_pretrain_cost(a, budget, proj):
    if os.path.exists(os.path.join(a.runs, "pretrain", "done")) or a.dry:
        return
    if proj["pretrain_cu"] > 0.5 * budget.left_cu():
        raise SystemExit(f"pretraining (~{proj['pretrain_cu']:.0f} CU) would take over half the remaining"
                         " budget; raise --budget_cu, or lower --pretrain_tokens after deleting data/ and runs/"
                         " both locally and in the Drive mirror")


def st_pretrain(a, budget):
    log = os.path.join(a.runs, "pretrain.log")
    out = os.path.join(a.runs, "pretrain")
    if os.path.exists(os.path.join(out, "done")):
        rec = load(os.path.join(out, "done"))
        print("pretrain: done earlier", json.dumps(rec))
        return check_valid(a, rec)
    cmd = [sys.executable, "pretrain.py", "--size", a.size, "--device", a.device, "--data_dir", a.data,
           "--out", out, "--batch", str(a.batch), "--lr", str(a.lr), "--warmup", "500",
           "--eval_every", "50" if a.dry else "2000"]
    if a.device.startswith("cuda"):
        cmd.append("--amp")
    print("+", " ".join(cmd), flush=True)
    with open(log, "a") as f:
        p = subprocess.Popen(cmd, cwd=ENGINE, stdout=f, stderr=subprocess.STDOUT)
        last = time.time()
        while p.poll() is None:
            time.sleep(5)
            if time.time() - last > 600:  # resume.pt to Drive every 10 minutes
                budget.save()
                mirror(a)
                last = time.time()
    if p.returncode:
        print(open(log).read()[-3000:])
        raise SystemExit("pretraining failed; rerun this stage to resume from the last checkpoint")
    rec = [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))][-1]
    print("pretrained:", json.dumps(rec), flush=True)
    open(os.path.join(out, "done"), "w").write(json.dumps(rec))
    mirror(a)  # model.pt and the marker reach Drive before the optimizer state is dropped
    for d in (out, os.path.join(a.mirror, "runs", "pretrain") if a.mirror else None):
        if d and os.path.exists(os.path.join(d, "resume.pt")):
            os.remove(os.path.join(d, "resume.pt"))  # optimizer state (~2 GB), not needed any more
    check_valid(a, rec)


def check_valid(a, rec):
    """Validity condition (DESIGN.md): the model must know the old facts before it can forget them."""
    if rec["old_acc_in"] < 0.80 and not a.dry:
        raise SystemExit(f"pretrained old-fact accuracy {rec['old_acc_in']:.3f} < 0.80: the testbed is"
                         " not valid at this scale; stop and report")


def st_prep(a, budget):
    """Build the caches every stream run reads (one process, so parallel runs never race)."""
    code = f"""
import json, torch, facts, stream
from tokenizers import Tokenizer
tok = Tokenizer.from_file({a.data + '/tokenizer.json'!r}); fj = json.load(open({a.data + '/facts.json'!r}))
m = stream.load_model({a.runs + '/pretrain/model.pt'!r}).to({a.device!r}); m.eval()
stream.background_df(m, {a.data + '/train.bin'!r}, 1000, {a.bg_windows}, {a.runs + f'/pretrain/bg_df_w{a.bg_windows}.pt'!r})
stream.protected_counts(m, tok, fj['old'][100:], {a.runs + '/pretrain/protect_noneval100_stream.pt'!r}, facts.STREAM_TEMPLATES)
stream.eval_doc_mask({a.data + '/train.bin'!r}, tok, [e['name'] for e in fj['old'][:100]], {a.data + '/eval_doc_mask100.npy'!r})
print('caches ready')
"""
    sh([sys.executable, "-c", code])


def sleep_sents(n_facts):
    """Sentences consolidated over a run's sleeps (stream.py: 7 per recent fact, 2 per older one)."""
    return sum(P_RECENT * SLEEP_EVERY + P_OLD * (n - SLEEP_EVERY) for n in range(SLEEP_EVERY, n_facts, SLEEP_EVERY))


def st_probe(a, budget):
    """Time 200-fact runs (one sleep) of each writer, then project the dev and test plan."""
    out = os.path.join(a.runs, "probe")
    if os.path.exists(os.path.join(a.runs, "projection.json")):
        return print("probe: done earlier; projection.json kept")
    n = 20 if a.dry else 200
    jobs = [stream_cmd(a, "C", 16, 0, out, n), stream_cmd(a, "p06ans", 4, 0, out, n),
            stream_cmd(a, "p06ans+gentle32", 4, 0, out, n), stream_cmd(a, "A", 1e-5, 0, out, n)]
    conc = min(a.workers, len(jobs))
    run_pool(a, budget, jobs, out, workers=conc)
    r = {nm.split("_lr")[0]: load(os.path.join(out, nm + ".json")) for nm, _ in jobs}
    wake = {k: r[k]["seconds"] / n for k in ("C", "p06ans", "A")}
    slept = sum(s["sleep_sents"] for s in r["p06ans+gentle32"]["sleeps"])
    per_sent = max(0.0, r["p06ans+gentle32"]["seconds"] - r["p06ans"]["seconds"]) / max(1, slept)
    est = {"C": wake["C"] * a.n_facts, "p06ans": wake["p06ans"] * a.n_facts, "A": wake["A"] * a.n_facts}
    est["C+gentle32"] = est["C"] + per_sent * sleep_sents(a.n_facts)
    est["p06ans+gentle32"] = est["p06ans"] + per_sent * sleep_sents(a.n_facts)
    plan = [arm for arm, lrs in DEV.items() for _ in lrs] + \
        [arm for arm in ("C", "p06ans", "C+gentle32", "p06ans+gentle32", "A") for _ in TEST_SEEDS]
    # Throughput measured at `conc` concurrent runs; more workers may be faster, never assumed.
    wall = sum(est[arm] for arm in plan) / conc
    proj = {"concurrency_measured": conc, "seconds_per_fact": wake, "sleep_seconds_per_sentence": per_sent,
            "est_run_minutes": {k: v / 60 for k, v in est.items()}, "runs_planned": len(plan),
            "projected_wall_minutes": wall / 60, "projected_cu": wall / 3600 * a.cu_per_hour,
            "spent_cu": budget.spent_cu(), "budget_cu": a.budget_cu}
    print(json.dumps(proj, indent=1), flush=True)
    if proj["projected_cu"] > budget.left_cu() and not a.dry:
        raise SystemExit("projected dev + test spend exceeds the remaining budget; raise --budget_cu"
                         " and rerun (finished work is kept)")
    json.dump(proj, open(os.path.join(a.runs, "projection.json"), "w"), indent=1)


def st_dev(a, budget):
    out = os.path.join(a.runs, "dev")
    jobs = [stream_cmd(a, arm, lr, 0, out, a.n_facts) for arm, lrs in DEV.items() for lr in lrs]
    run_pool(a, budget, jobs, out)


def best_lr(dev, arm, tie=0.02):
    """Highest retention; within `tie` of it, the lowest old-fact drop (the Exp 02 tie rule)."""
    sys.path.insert(0, ENGINE)
    from analyze import run_metrics  # noqa: E402
    ms = {lr: run_metrics(load(os.path.join(dev, f"{arm}_lr{lr:g}_s0.json"))) for lr in DEV[arm]}
    best = max(m["ret_in"] for m in ms.values())
    near = [(lr, m) for lr, m in ms.items() if best - m["ret_in"] <= tie]
    lr = min(near, key=lambda t: t[1]["old_drop"])[0]
    return lr, {f"{k:g}": {"ret_in": m["ret_in"], "old_drop": m["old_drop"], "cap_frac": m["cap_frac"]}
                for k, m in ms.items()}


def st_freeze(a, budget):
    out = os.path.join(a.runs, "frozen.json")
    if os.path.exists(out):
        return print("frozen:", open(out).read())
    dev = os.path.join(a.runs, "dev")
    sys.path.insert(0, EXP02)
    import freeze  # noqa: E402  (Exp 02's eligibility rule, unchanged)
    rep = freeze.fu1(dev)
    if isinstance(rep.get("L*"), float):
        L, flag = rep["L*"], rep.get("flag")
    else:  # no LR passes even the thin-margin rule: test the best one anyway and report the failure
        L, _ = best_lr(dev, "p06ans", tie=0.0)
        flag = "no eligible LR at this scale (" + str(rep.get("L*", rep.get("decision"))) + ")"
    c_lr, c_rep = best_lr(dev, "C")
    a_lr, a_rep = best_lr(dev, "A")
    frozen = {"L*": L, "flag": flag, "C_lr": c_lr, "A_lr": a_lr,
              "R": "gentle32 (frozen in Exp 02, not re-selected)",
              "fu1_report": rep["report"], "C_report": c_rep, "A_report": a_rep}
    json.dump(frozen, open(out, "w"), indent=1, default=str)
    print(json.dumps(frozen, indent=1, default=str), flush=True)


def st_test(a, budget):
    f = load(os.path.join(a.runs, "frozen.json"))
    out = os.path.join(a.runs, "test")
    jobs = []  # sleep arms first: they are the longest
    for arm, lr in (("p06ans+gentle32", f["L*"]), ("C+gentle32", f["C_lr"]), ("A", f["A_lr"]),
                    ("p06ans", f["L*"]), ("C", f["C_lr"])):
        jobs += [stream_cmd(a, arm, lr, s, out, a.n_facts) for s in TEST_SEEDS]
    run_pool(a, budget, jobs, out)


def st_analyze(a, budget):
    L = load(os.path.join(a.runs, "frozen.json"))["L*"]
    res = os.path.join(a.runs, "results")
    sh([sys.executable, os.path.join(EXP02, "analyze02.py"), "--runs", os.path.join(a.runs, "test"),
        "--seeds", *map(str, TEST_SEEDS), "--primary", f"p06ans:{L:g}", "--extra_runs", "--out", res],
       cwd=HERE, log=os.path.join(a.runs, "analyze.log"))
    print(open(os.path.join(a.runs, "analyze.log")).read()[-6000:], flush=True)


def st_package(a, budget):
    """Copy everything small into the repo checkout (exp03_scale/results_colab) and tar it."""
    dst = os.path.join(HERE, "results_colab")
    os.makedirs(dst, exist_ok=True)
    for sub in ("dev", "test", "probe", "results"):
        src = os.path.join(a.runs, sub)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(dst, sub), dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.pt", "*.npy"))
    for f in ("budget.json", "projection.json", "frozen.json", "smoke_ok.json", "data.log", "analyze.log"):
        if os.path.exists(os.path.join(a.runs, f)):
            shutil.copy(os.path.join(a.runs, f), dst)
    if os.path.exists(os.path.join(a.runs, "pretrain", "log.jsonl")):
        shutil.copy(os.path.join(a.runs, "pretrain", "log.jsonl"), os.path.join(dst, "pretrain_log.jsonl"))
    tar = os.path.join(a.runs, "exp03_results.tar.gz")
    sh(["tar", "czf", tar, "-C", HERE, "results_colab"], cwd=HERE)
    print("packaged:", tar, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", help="one of %s, or all" % STAGES)
    ap.add_argument("--data", default="/content/exp03/data", help="local disk (fast); mirrored to --mirror")
    ap.add_argument("--runs", default="/content/exp03/runs")
    ap.add_argument("--mirror", default="", help="Drive folder to copy data and runs to, and restore from")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--size", default="small")
    ap.add_argument("--pretrain_tokens", type=int, default=300_000_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--n_old", type=int, default=1000)
    ap.add_argument("--bios_per_old", type=int, default=100)
    ap.add_argument("--n_facts", type=int, default=2000)
    ap.add_argument("--bg_windows", type=int, default=16, help="16 x 256 tokens = Meta's 4,096-token batches")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget_cu", type=float, default=60)
    ap.add_argument("--cu_per_hour", type=float, default=13.0, help="read it off Colab's Resources panel")
    ap.add_argument("--dry", action="store_true", help="tiny end-to-end check (CPU ok)")
    a = ap.parse_args()
    if a.dry:
        a.size, a.pretrain_tokens, a.n_old, a.bios_per_old, a.batch = "tiny", 600_000, 120, 10, 16
        a.n_facts, a.workers, a.bg_windows = 40, 2, 2
        DEV.update({"C": [16, 32], "p06ans": [4, 8], "A": [1e-5]})
    a.data, a.runs = os.path.abspath(a.data), os.path.abspath(a.runs)
    if a.mirror:  # restore whatever an earlier session left on Drive
        sync(os.path.join(a.mirror, "data"), a.data)
        sync(os.path.join(a.mirror, "runs"), a.runs)
    os.makedirs(a.runs, exist_ok=True)
    os.makedirs(a.data, exist_ok=True)
    budget = Budget(a)
    for st in STAGES if a.stage == "all" else [a.stage]:
        print(f"\n===== {st} =====", flush=True)
        t0 = time.time()
        globals()["st_" + st](a, budget)
        budget.add(st, time.time() - t0)
        mirror(a)
        budget.report()


if __name__ == "__main__":
    main()
