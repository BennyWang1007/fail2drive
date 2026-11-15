#!/usr/bin/env python3

import argparse
import json
import re
import shutil
from pathlib import Path


INFRACTION_KEYS = [
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "min_speed_infractions",
    "yield_emergency_vehicle_infractions",
    "scenario_timeouts",
    "route_dev",
    "vehicle_blocked",
    "route_timeout",
]

CATEGORY_MAP = {
    "fail": "Fail",
    "not_ideal": "Not ideal",
    "not-ideal": "Not ideal",
    "slow": "Slow",
}


def has_infraction(value):
    """Return True if an infraction value is nonzero or nonempty."""
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, list):
        return len(value) > 0
    return bool(value)


def classify_result(path: Path) -> str:
    """Classify a single result JSON as Fail, Not ideal, Slow or Perfect."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        checkpoint = data.get("_checkpoint", {})
        global_record = checkpoint.get("global_record", {})
        infractions = global_record.get("infractions", {})

        status = global_record.get("status", "")

        if status == "Perfect":
            return "Perfect"

        # Failed, incomplete, or invalid runs.
        if (
            data.get("entry_status") != "Finished"
            or not data.get("eligible", False)
            or status != "Completed"
            or global_record.get("meta", {}).get("exceptions")
        ):
            return "Fail"

        # Treat missing infraction fields as zero.
        values = {
            key: infractions.get(key, 0)
            for key in INFRACTION_KEYS
        }

        # Slow: min-speed infractions are the only infractions.
        min_speed = values.pop("min_speed_infractions")

        # print(f"DEBUG: {path.name} infractions: {values}, min_speed={min_speed}")

        if has_infraction(min_speed) and not any(
            has_infraction(value) for value in values.values()
        ):
            return "Slow"

        # Not ideal: any other infractions, or no infractions at all.
        return "Not ideal"

    except (OSError, json.JSONDecodeError, TypeError, AttributeError) as e:
        print(f"Warning: cannot parse {path.name}: {e}")
        return "Fail"


def is_valid_result(res_path: Path) -> bool:
    """Check if the result JSON exists, is valid JSON, and represents a valid run."""
    if not res_path or not res_path.is_file():
        return False
    try:
        with res_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        checkpoint = data.get("_checkpoint", {})
        global_record = checkpoint.get("global_record", {})
        if not global_record:
            return False
        return True
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
        return False


def get_score(res_path: Path) -> tuple:
    """
    Extract driving score from *_res.json for comparison.
    Returns a tuple: (is_finished_and_completed, score_composed, score_route, -infraction_count)
    so Python's max() naturally picks the best run even when scores tie.
    """
    try:
        with res_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        checkpoint = data.get("_checkpoint", {})
        global_record = checkpoint.get("global_record", {})

        # Fallback to records[0] if global_record scores are empty
        records = checkpoint.get("records", [])
        rec = global_record if global_record.get("scores") else (records[0] if records else {})

        scores = rec.get("scores", global_record.get("scores_mean", {}))
        score_composed = float(scores.get("score_composed", -1.0))
        score_route = float(scores.get("score_route", -1.0))
        score_penalty = float(scores.get("score_penalty", -1.0))

        is_completed = int(
            data.get("entry_status") == "Finished"
            and bool(data.get("eligible", False))
            and global_record.get("status") == "Completed"
        )

        return (is_completed, score_composed, score_route, score_penalty)
    except Exception:
        return (-1, -1.0, -1.0, -1.0)


def resolve_dirs(input_dir: Path) -> tuple[Path, Path]:
    """
    Resolve (base_dir, res_dir) from user input.
    Supports passing either `.../hipad_oracle/3` or `.../hipad_oracle/3/res`.
    """
    if not input_dir.is_dir():
        raise ValueError(f"Not a directory: {input_dir}")

    # Case 1: User passed `.../3` which contains a `res/` folder
    if (input_dir / "res").is_dir():
        return input_dir, input_dir / "res"

    # Case 2: User passed `.../3/res` directly
    if input_dir.name == "res" and list(input_dir.glob("*_res.json")):
        return input_dir.parent, input_dir

    # Case 3: *_res.json files are directly in input_dir
    return input_dir, input_dir


def find_unique_removed_dir(removed_root: Path, route_id: str) -> Path:
    """Return `removed/route_id`, or `removed/route_id_2`, `_3`, etc. if it already exists."""
    candidate = removed_root / route_id
    if not candidate.exists():
        return candidate

    idx = 2
    while True:
        candidate = removed_root / f"{route_id}_{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def find_file_in_workspace(base_dir: Path, filename: str, subdirs: list[str]) -> Path | None:
    """Locate a file in base_dir or common subdirectories (excluding 'removed')."""
    direct = base_dir / filename
    if direct.is_file():
        return direct

    for sub in subdirs:
        candidate = base_dir / sub / filename
        if candidate.is_file():
            return candidate

    # Fallback recursive search (ignoring removed/)
    for match in base_dir.rglob(filename):
        if "removed" not in match.relative_to(base_dir).parts and match.is_file():
            return match
    return None


def resolve_restore_target(base_dir: Path, filename: str, candidate_subdirs: list[str]) -> Path:
    """
    Determine where to restore a file (e.g., xxxx.err.log or eval_xxxx.sh).
    If a matching subdirectory exists in base_dir, place it there; otherwise place in base_dir.
    """
    for sub in candidate_subdirs:
        sub_path = base_dir / sub
        if sub_path.is_dir():
            return sub_path / filename
    return base_dir / filename


def move_route_to_removed(
    base_dir: Path,
    res_dir: Path,
    route_id: str,
    dry_run: bool = False,
) -> Path:
    """
    Move all artifacts of `route_id` into `base_dir/removed/<route_id>[_N]/`:
      - xxxx.err.log
      - xxxx.out.log
      - xxxx_res.json
      - eval_xxxx.sh
      - viz/ (containing the contents of base_dir/viz/xxxx/*)
    """
    removed_root = base_dir / "removed"

    if dry_run:
        target_dir = find_unique_removed_dir(removed_root, route_id)
    else:
        removed_root.mkdir(parents=True, exist_ok=True)
        target_dir = find_unique_removed_dir(removed_root, route_id)
        target_dir.mkdir(parents=True, exist_ok=True)

    items_to_move: list[Path] = []

    # 1. Move xxxx_res.json
    res_file = res_dir / f"{route_id}_res.json"
    if not res_file.is_file():
        res_file = find_file_in_workspace(base_dir, f"{route_id}_res.json", ["res", "results"])
    if res_file and res_file.is_file():
        # shutil.move(str(res_file), str(target_dir / res_file.name))
        # moved_items.append(res_file.name)
        items_to_move.append(res_file)

    # 2. Move xxxx.err.log and xxxx.out.log
    # for ext in ["err.log", "out.log"]:
    for suffix in ["err.log", "out.log", "slurm.err", "slurm.out"]:
        fname = f"{route_id}_{suffix}"
        log_file = find_file_in_workspace(base_dir, fname, ["logs", "err", "out", "slurm_out", "slurm_err"])
        if log_file and log_file.is_file():
            # shutil.move(str(log_file), str(target_dir / fname))
            # moved_items.append(fname)
            items_to_move.append(log_file)

    # 3. Move eval_xxxx.sh
    sh_name = f"eval_{route_id}.sh"
    sh_file = find_file_in_workspace(base_dir, sh_name, ["scripts", "sh", "eval", "slurm", "jobs"])
    if sh_file and sh_file.is_file():
        # shutil.move(str(sh_file), str(target_dir / sh_name))
        # moved_items.append(sh_name)
        items_to_move.append(sh_file)

    # 4. Move viz/xxxx/* into target_dir/viz/
    viz_src = base_dir / "viz" / route_id
    if viz_src.is_dir():
        # viz_dst = target_dir / "viz"
        # shutil.move(str(viz_src), str(viz_dst))
        # moved_items.append(f"viz/ (from viz/{route_id})")
        items_to_move.append(viz_src)

    # 5. Move debug/xxxx/* into target_dir/debug/
    debug_src = base_dir / "debug" / route_id
    if debug_src.is_dir():
        items_to_move.append(debug_src)

    # Print or execute
    if dry_run:
        print(f"[Dry-run] '{route_id}' -> {target_dir.relative_to(base_dir)}/")
        for item in items_to_move:
            print(f"  {item.relative_to(base_dir)}{'/' if item.is_dir() else ''}")
    else:
        for item in items_to_move:
            if item == debug_src:
                dst = target_dir / "debug"
            elif item == viz_src:
                dst = target_dir / "viz"
            else:
                dst = target_dir / item.name
            shutil.move(str(item), str(dst))

        print(
            f"[Remove] '{route_id}' -> "
            f"{target_dir.relative_to(base_dir)} "
            f"({', '.join(item.name for item in items_to_move)})"
        )

    # print(f"[Remove] '{route_id}' -> {target_dir.relative_to(base_dir)} ({', '.join(moved_items)})")
    return target_dir


def recover_route(base_dir: Path, res_dir: Path, route_id: str):
    """
    Find the highest-scoring backup in `removed/route_id[_%d]` and restore it to top-level.
    If a valid result currently exists in the workspace, move it to `removed/` first.
    """
    removed_root = base_dir / "removed"
    if not removed_root.is_dir():
        print(f"[Recover] Error: '{removed_root}' does not exist.")
        return

    # Match removed/<route_id> or removed/<route_id>_<N>
    pattern = re.compile(rf"^{re.escape(route_id)}(?:_(\d+))?$")
    candidates = [
        d for d in removed_root.iterdir()
        if d.is_dir() and pattern.match(d.name)
    ]

    if not candidates:
        print(f"[Recover] No removed backups found for route '{route_id}' in {removed_root}")
        return

    # Rank candidates by score in xxxx_res.json
    scored_candidates = []
    for cand in candidates:
        cand_res = cand / f"{route_id}_res.json"
        score_tuple = get_score(cand_res) if cand_res.is_file() else (-1, -1.0, -1.0, -1.0)
        scored_candidates.append((score_tuple, cand))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_dir = scored_candidates[0]

    print(
        f"[Recover] Best backup for '{route_id}' is '{best_dir.name}' "
        f"(composed={best_score[1]:.2f}, route={best_score[2]:.2f}, penalty={best_score[3]:.2f})"
    )

    # Check if current top-level has an active result; if valid, archive it to removed first
    current_res = res_dir / f"{route_id}_res.json"
    current_viz = base_dir / "viz" / route_id
    if is_valid_result(current_res):
        print(f"[Recover] Current result for '{route_id}' is valid. Moving it to removed/ first...")
        move_route_to_removed(base_dir, res_dir, route_id)
    else:
        # Clean up any leftover invalid/partial files before restoring
        if current_res.is_file():
            current_res.unlink()
        if current_viz.is_dir():
            shutil.rmtree(current_viz)

    # Restore files from best_dir back to top-level structure
    restored_items = []

    # 1. Restore xxxx_res.json -> res_dir/xxxx_res.json
    res_src = best_dir / f"{route_id}_res.json"
    if res_src.is_file():
        res_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(res_src), str(res_dir / res_src.name))
        restored_items.append(str((res_dir / res_src.name).relative_to(base_dir)))

    # 2. Restore xxxx.err.log and xxxx.out.log
    for ext in ["err.log", "out.log"]:
        fname = f"{route_id}.{ext}"
        src = best_dir / fname
        if src.is_file():
            dst = resolve_restore_target(base_dir, fname, ["logs", "log", "out", "slurm"])
            shutil.move(str(src), str(dst))
            restored_items.append(str(dst.relative_to(base_dir)))

    # 3. Restore eval_xxxx.sh
    sh_name = f"eval_{route_id}.sh"
    sh_src = best_dir / sh_name
    if sh_src.is_file():
        dst = resolve_restore_target(base_dir, sh_name, ["scripts", "sh", "eval", "slurm", "jobs"])
        shutil.move(str(sh_src), str(dst))
        restored_items.append(str(dst.relative_to(base_dir)))

    # 4. Restore viz/ -> base_dir/viz/xxxx/
    viz_src = best_dir / "viz"
    if viz_src.is_dir():
        viz_dst = base_dir / "viz" / route_id
        viz_dst.parent.mkdir(parents=True, exist_ok=True)
        if viz_dst.exists():
            shutil.rmtree(viz_dst)
        shutil.move(str(viz_src), str(viz_dst))
        restored_items.append(str(viz_dst.relative_to(base_dir)))

    # Remove the backup directory if it is now empty
    try:
        best_dir.rmdir()
    except OSError:
        pass

    print(f"[Recover] Restored '{route_id}' from {best_dir.name} -> {', '.join(restored_items)}")


def parse_remove_categories(raw_list: list[str] | None) -> set[str]:
    """Normalize user-specified `-r` categories into {'Fail', 'Not ideal', 'Slow'}."""
    if not raw_list:
        return set()
    selected = set()
    for item in raw_list:
        key = item.strip().lower().replace(" ", "_")
        if key not in CATEGORY_MAP:
            valid = ", ".join(sorted(CATEGORY_MAP.keys()))
            raise ValueError(f"Unknown category '{item}'. Valid choices: {valid}")
        selected.add(CATEGORY_MAP[key])
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Classify, remove, or recover evaluation result files."
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        help="Experiment root directory (e.g., results_shome/hipad_oracle/3) or res directory",
    )
    parser.add_argument(
        "-r",
        "--remove",
        action="append",
        metavar="CATEGORY",
        help="Category to move to removed/ (can be specified multiple times: fail, not_ideal, slow)",
    )
    parser.add_argument(
        "--recover",
        nargs="+",
        metavar="ROUTE_ID",
        help="Route ID(s) to recover the highest-scoring run from removed/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files and directories that would be moved without modifying anything.",
    )
    args = parser.parse_args()
    dry_run = args.dry_run

    try:
        base_dir, res_dir = resolve_dirs(args.result_dir)
    except ValueError as e:
        parser.error(str(e))

    # Handle --recover mode
    if args.recover:
        for route_id in args.recover:
            recover_route(base_dir, res_dir, route_id)
        if not args.remove:
            return

    try:
        remove_categories = parse_remove_categories(args.remove)
    except ValueError as e:
        parser.error(str(e))

    files = sorted(res_dir.glob("*_res.json"))
    if not files:
        parser.error(f"No *_res.json files found in {res_dir}")

    categories = {
        "Fail": [],
        "Not ideal": [],
        "Slow": [],
        "Perfect": [],
    }

    routes_to_remove = []

    for path in files:
        cat = classify_result(path)
        route_id = path.stem.removesuffix("_res")
        categories[cat].append(route_id)
        if cat in remove_categories:
            routes_to_remove.append((route_id, cat))

    for category, runs in categories.items():
        print(f"{category}: {len(runs)} runs")
        print(", ".join(runs))
        print()

    if routes_to_remove:
        print(f"--- Moving {len(routes_to_remove)} runs to {base_dir / 'removed'} ---")
        for route_id, cat in routes_to_remove:
            move_route_to_removed(base_dir, res_dir, route_id, dry_run=dry_run)


if __name__ == "__main__":
    main()
