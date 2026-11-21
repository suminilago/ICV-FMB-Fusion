"""runs/*/log.jsonl, args.json 만으로 만들 수 있는 표와 그림을 results/ 에 다시 만든다.

데이터, 체크포인트, GPU 가 필요 없다. 레포만 clone 해서 실행하면 된다.

    python tools/analyze_runs.py
    python tools/analyze_runs.py --out /tmp/check      # 레포를 건드리지 않고 확인

epoch 는 모두 1부터 센다 (log.jsonl 의 epoch + 1). 쌍별 격차는 (앞 run) - (뒤 run).
log.jsonl 이 없는 run 은 건너뛰고 알려 준다.

만드는 파일 (results/ 기준)
  00_overview/run_summary.csv       run 별 설정, best val mIoU, best epoch
  00_overview/run_convergence.csv   epoch 20 -> 마지막 상승폭, best 가 마지막 epoch 인지
  00_overview/curve_all_runs.png    전체 학습 곡선 (B0 / B2)
  <사이클>/run_gap_<a>_vs_<b>.csv   epoch 별 val mIoU 와 격차
  <사이클>/curve_gap_<a>_vs_<b>.png 위: 두 곡선, 아래: epoch 별 격차
"""
import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # 그림만 생략한다
    plt = None

ROOT = Path(__file__).resolve().parents[1]

# 11개 실험의 run 폴더 (docs/experiment_design.md 와 results/report.md 의 번호 순서)
EXPECTED = [
    "fmb_rgb", "fmb_sum", "fmb_sum_oversample10", "fmb_concat", "fmb_adaptive",
    "fmb_concat_res", "fmb_early", "fmb_b2_sum", "fmb_b2_sum_480",
    "fmb_b2_rgb_480", "fmb_b2_sum_480_oversample10",
]

# (사이클 폴더, 앞 run, 뒤 run): 앞 - 뒤 를 epoch 별로 본다
PAIRS = [
    ("01_baseline", "fmb_sum", "fmb_rgb"),
    ("01_baseline", "fmb_sum_oversample10", "fmb_sum"),
    ("02_fusion", "fmb_concat_res", "fmb_concat"),
    ("02_fusion", "fmb_adaptive", "fmb_concat"),
    ("03_capacity", "fmb_b2_sum", "fmb_sum"),
    ("03_capacity", "fmb_b2_sum_480", "fmb_b2_sum"),
    ("04_ir_contribution", "fmb_b2_sum_480", "fmb_b2_rgb_480"),
    ("05_oversampling_final", "fmb_b2_sum_480_oversample10", "fmb_b2_sum_480"),
]

# 모든 run 에서 같아야 하는 고정 조건
FIXED = ["epochs", "bs", "lr", "wd", "warmup", "seed"]


def short(run):
    return run[4:] if run.startswith("fmb_") else run


def load_runs(runs_dir):
    runs = {}
    for d in sorted(runs_dir.iterdir()):
        log, args = d / "log.jsonl", d / "args.json"
        if not (d.is_dir() and log.exists() and args.exists()):
            continue
        rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        runs[d.name] = {
            "args": json.loads(args.read_text()),
            "val": [r["val_mIoU"] for r in rows],
            "epoch": [r["epoch"] + 1 for r in rows],
        }
    return runs


def best(run):
    v = run["val"]
    i = max(range(len(v)), key=v.__getitem__)
    return v[i], run["epoch"][i]


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def summary(runs, out):
    rows = []
    for name, r in runs.items():
        a = r["args"]
        b, be = best(r)
        rows.append([
            name, a["fusion"], a["backbone"].split("/")[-1], a["height"], a["width"],
            a["epochs"], a["seed"], a.get("target_fraction", ""),
            round(b, 2), be, len(r["val"]), round(r["val"][-1], 2),
        ])
    write_csv(out / "00_overview" / "run_summary.csv",
              ["run", "fusion", "backbone", "height", "width", "epochs", "seed",
               "target_fraction", "best_val_mIoU", "best_epoch", "logged_epochs",
               "final_val_mIoU"], rows)


def convergence(runs, out):
    rows = []
    for name, r in runs.items():
        v = r["val"]
        b, be = best(r)
        rows.append([name, be, len(v), be == len(v), round(v[9], 2), round(v[19], 2),
                     round(v[-1], 2), round(v[-1] - v[19], 2)])
    write_csv(out / "00_overview" / "run_convergence.csv",
              ["run", "best_epoch", "logged_epochs", "best_is_last", "val_ep10",
               "val_ep20", "val_last", "gain_ep20_to_last"], rows)


def curves(runs, out):
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, bb in zip(axes, ("mit-b0", "mit-b2")):
        for name, r in runs.items():
            if r["args"]["backbone"].endswith(bb):
                ax.plot(r["epoch"], r["val"], label=short(name))
        ax.set_title("B0 (384x512)" if bb == "mit-b0" else "B2")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val mIoU (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "00_overview" / "curve_all_runs.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)


def gaps(runs, out):
    for cycle, a, b in PAIRS:
        if a not in runs or b not in runs:
            continue
        va, vb = runs[a]["val"], runs[b]["val"]
        n = min(len(va), len(vb))
        gap = [va[i] - vb[i] for i in range(n)]
        name = f"{short(a)}_vs_{short(b)}"
        write_csv(out / cycle / f"run_gap_{name}.csv",
                  ["epoch", short(a), short(b), "gap"],
                  [[i + 1, round(va[i], 2), round(vb[i], 2), round(gap[i], 2)]
                   for i in range(n)])
        m = max(range(n), key=gap.__getitem__)
        print(f"  {cycle}/{name}: 최대 격차 {gap[m]:+.2f} (epoch {m + 1}), "
              f"마지막 {gap[-1]:+.2f}, 최소 {min(gap):+.2f}")
        if plt is None:
            continue
        x = list(range(1, n + 1))
        fig, (t, g) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
        t.plot(x, va[:n], label=short(a))
        t.plot(x, vb[:n], label=short(b))
        t.set_ylabel("val mIoU (%)")
        t.legend()
        t.grid(alpha=0.3)
        g.bar(x, gap, color=["tab:green" if d >= 0 else "tab:red" for d in gap])
        g.axhline(0, color="k", lw=0.8)
        g.set_ylabel("gap (pp)")
        g.set_xlabel("epoch")
        g.grid(alpha=0.3)
        fig.suptitle(f"{short(a)} - {short(b)}")
        fig.tight_layout()
        fig.savefig(out / cycle / f"curve_gap_{name}.png", dpi=150)
        plt.close(fig)


def check_fixed(runs):
    for k in FIXED:
        vals = {json.dumps(r["args"].get(k)) for r in runs.values()}
        if len(vals) > 1:
            print(f"경고: run 사이에 {k} 가 다르다: {sorted(vals)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    runs_dir, out = Path(a.runs), Path(a.out)
    if not runs_dir.is_dir():
        sys.exit(f"{runs_dir} 가 없다")

    runs = load_runs(runs_dir)
    print(f"log.jsonl 을 읽은 run {len(runs)}개")
    missing = [r for r in EXPECTED if r not in runs]
    if missing:
        print("로그가 없어 건너뛴 run:", ", ".join(missing))
    if not runs:
        sys.exit("읽을 run 이 없다")
    if plt is None:
        print("matplotlib 이 없어 그림은 만들지 않는다")

    check_fixed(runs)
    summary(runs, out)
    convergence(runs, out)
    curves(runs, out)
    print("epoch 별 격차:")
    gaps(runs, out)
    print(f"출력: {out}")


if __name__ == "__main__":
    main()
