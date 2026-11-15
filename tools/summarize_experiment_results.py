#!/usr/bin/env python3
"""Refresh the Markdown snapshot using parres, without submitting experiments."""

import math
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from f2d_result_parser import (
    build_main_table,
    build_route_mapping,
    harmonic_mean,
    load_rows,
    resolve_seed_dirs,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_all_evals import TASKS  # Importing only reads task configuration.

BEGIN = "<!-- BEGIN PARRES SNAPSHOT -->"
END = "<!-- END PARRES SNAPSHOT -->"


def metrics(rows):
    table = build_main_table(rows)
    values = []
    for split in ("Base", "Generalization"):
        selected = table[table["Split"] == split]
        if selected.empty:
            values.extend(["—"] * 3)
            continue
        ds = float(selected.iloc[0]["DS"])
        success = float(selected.iloc[0]["Success"])
        values.extend(
            f"{value:.1f}" if math.isfinite(value) else "—"
            for value in (ds, success, harmonic_mean(ds, success))
        )
    return values


def main():
    started = datetime.now(ZoneInfo("Asia/Taipei"))
    catalog_text = (ROOT / "notes/experiment.sh").read_text()
    # Read the literal catalog, never source or execute the submission script.
    catalog = re.findall(r"^([a-z0-9_]+)\|([1-3](?:,[1-3])*)$", catalog_text, re.M)
    if not catalog:
        raise SystemExit("No experiment catalog entries found")
    experiments = [(name, seeds.split(",")) for name, seeds in catalog]
    # Include reproduction baselines alongside the submission catalog.
    # experiments.extend((name, ["1", "2", "3"]) for name in
    #                    ("tfv6_no_policy", "hipad_no_policy") if name not in dict(catalog))
    route_map = build_route_mapping(ROOT / "fail2drive_split")
    expected_ids = {f"{idx:04d}" for idx in route_map}
    blocks = []
    total_missing = 0
    total_valid = 0
    for name, seeds in experiments:
        config = TASKS[name]
        result_root = ROOT / config["resolved_out_root"]
        directories = resolve_seed_dirs([str(result_root / seed / "res") for seed in seeds])
        rows, warnings, _ = load_rows(name, directories, route_map)
        missing_counts = []
        valid = 0
        for seed, path in directories:
            selected = rows if rows.empty else rows[rows["Seed"] == seed]
            ids = set() if selected.empty else set(selected["Id"])
            missing = len(expected_ids - ids)
            missing_counts.append(f"seed{seed}: {missing}")
            total_missing += missing
            total_valid += len(ids)
            valid += len(ids)
        key_warnings = [warning for warning in warnings
                        if re.match(r"WARNING: seed .+ is missing \d+ results:", warning)
                        or "directory does not exist:" in warning]
        blocks.extend([
            f"### {name} ({valid}/{len(expected_ids) * len(seeds)})",
            "",
            "```text",
            *key_warnings,
            "scores: " + "\t".join(metrics(rows)),
            "```",
            "",
        ])
        print(f"{name}: {', '.join(missing_counts)}; parsed rows={len(rows)}", flush=True)

    ended = datetime.now(ZoneInfo("Asia/Taipei"))
    lines = [
        BEGIN,
        "## 最新 parres 統計",
        "",
        f"讀取時間：{started:%Y-%m-%d %H:%M:%S} ～ {ended:%H:%M:%S}（台灣時間）。",
        f"範圍：`notes/experiment.sh` 的 {len(catalog)} 種實驗、"
        f"{sum(len(seeds.split(',')) for _, seeds in catalog)} 組 task/seed，另加兩個 reproduced baseline 的 seeds 1、2、3。",
        f"每個 seed 預期 {len(expected_ids)} 條 route；合計可計分 {total_valid} 條，缺少 {total_missing} 條。",
        "",
        "缺少 = XML 清單中的 route 尚無 parres 可計分的 record；缺目錄、無 record、agent setup 失敗、缺分數均計入。",
        "這是結果統計，不代表 Slurm job 是否完成；失敗 route 若已有可計分 record，仍沿用 parres 算法納入。",
        "六個分數為 Base／Generalization 各自的 DS、Success（%）、HM；HM 為 DS 與 Success 的調和平均。",
        "Success 沿用 parres 的 infractions 判定，忽略 min_speed_infractions 與 outside_route_lanes。",
        "分數只平均現有有效結果；缺資料不補零，完全沒有該 split 的資料顯示「—」。未齊全的數字是暫時結果，執行中的結果可能在讀取期間更新。",
        "",
        "合併分數直接平均所有有效 route，與 parres 多目錄輸入一致。",
        "`scores:` 後依序為 Base DS、Base Success、Base HM、Gen DS、Gen Success、Gen HM；數字之間使用實際 Tab 字元。複製六個數字（不含 `scores:`）可貼入試算表的六欄。",
        "",
        *blocks,
        "重新整理：`/mnt/bapve/thome/wangyuhao/miniconda3/envs/fail2drive/bin/python tools/summarize_experiment_results.py`。此指令只分析結果並更新本區塊。",
        END,
    ]
    destination = ROOT / "notes/eval_results.md"
    content = destination.read_text()
    block = "\n".join(lines)
    if BEGIN in content:
        start = content.index(BEGIN)
        end = content.index(END, start) + len(END)
        content = content[:start] + block + content[end:]
    else:
        anchor = "<!-- END EXPERIMENT SUBMISSIONS -->"
        if anchor in content:
            content = content.replace(anchor, anchor + "\n\n" + block, 1)
        else:
            content = block + "\n\n" + content
    destination.write_text(content)
    print(f"Wrote {destination}: valid={total_valid}, missing={total_missing}", flush=True)


if __name__ == "__main__":
    main()
