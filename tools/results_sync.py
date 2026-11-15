
#!/usr/bin/env python3

from pathlib import Path, PurePosixPath
import argparse
import tempfile
import zipfile
import shutil
import os
import re

# ============================================================
# Configuration
# ============================================================

BASE = Path.home() / "fail2drive"
ZIP_FILE = BASE / "results_collected.zip"

SEEDS = ("1", "2", "3")

RESULT_TYPES = ("res",)
ALL_RESULT_TYPES = ("res", "err", "out", "viz")

EXCLUDE = re.compile(
    r"^(alpalog|.*log|logs|server.*|slurm.*|.*probe|tfv6|quick_.*)$"
)


# ============================================================
# Helpers
# ============================================================

def get_result_types(all_mode):
    return ALL_RESULT_TYPES if all_mode else RESULT_TYPES


def get_experiments(directory):
    if not directory.is_dir():
        return []

    return sorted(
        p for p in directory.iterdir()
        if p.is_dir() and not EXCLUDE.match(p.name)
    )


def valid_archive_path(name, result_types):
    parts = PurePosixPath(name).parts

    if len(parts) < 4:
        return None

    if parts[0] not in ("results_shome", "results_collected"):
        return None

    if parts[1] in ("", ".", "..") or EXCLUDE.match(parts[1]):
        return None

    if parts[2] not in SEEDS:
        return None

    if parts[3] not in result_types:
        return None

    if any(p in ("", ".", "..") for p in parts):
        return None

    return Path(*parts[1:])


def add_file(archive, file, arcname):
    archive.write(file, arcname)


# ============================================================
# COLLECT
# ============================================================

def collect(output_dir, all_mode=False):

    result_types = get_result_types(all_mode)
    source = BASE / output_dir

    if not source.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")

    print("=" * 60)
    print("COLLECT")
    print("=" * 60)
    print(f"Local source : {source}")
    print(f"Input ZIP    : {ZIP_FILE}")
    print(f"Output ZIP   : {ZIP_FILE}")
    print(f"Types        : {', '.join(result_types)}")
    print()

    ZIP_FILE.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix="results_collected_",
        suffix=".zip",
        dir=BASE
    )
    os.close(fd)

    temp_zip = Path(temp_name)

    archive_files = 0
    local_files = 0

    try:
        with zipfile.ZipFile(
            temp_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6
        ) as output:

            # ------------------------------------------------
            # 1. Copy previous ZIP directly (baseline)
            # ------------------------------------------------

            if ZIP_FILE.is_file():

                print("=== Merging previous ZIP ===")

                with zipfile.ZipFile(ZIP_FILE, "r") as previous:

                    for info in previous.infolist():

                        if info.is_dir():
                            continue

                        relative = valid_archive_path(
                            info.filename,
                            result_types
                        )

                        if relative is None:
                            continue

                        arcname = Path("results_collected") / relative

                        output.writestr(
                            arcname.as_posix(),
                            previous.read(info)
                        )

                        archive_files += 1

                print(f"  Imported: {archive_files} files")

            # ------------------------------------------------
            # 2. Add local files (local takes priority)
            # ------------------------------------------------

            print(f"\n=== Adding local {output_dir} ===")

            for experiment in get_experiments(source):

                for seed in SEEDS:

                    for result_type in result_types:

                        directory = (
                            experiment / seed / result_type
                        )

                        if not directory.is_dir():
                            continue

                        for file in sorted(directory.rglob("*")):

                            if not file.is_file():
                                continue

                            relative = (
                                Path(experiment.name)
                                / seed
                                / result_type
                                / file.relative_to(directory)
                            )

                            arcname = (
                                Path("results_collected") / relative
                            )

                            add_file(output, file, arcname)
                            local_files += 1

            print(f"  Added: {local_files} local files")

        # Replace old ZIP only after successful completion.
        os.replace(temp_zip, ZIP_FILE)

    finally:
        if temp_zip.exists():
            temp_zip.unlink()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("COLLECT COMPLETE")
    print("=" * 60)
    print(f"Previous ZIP files : {archive_files}")
    print(f"Local files        : {local_files}")
    print(f"Result types       : {', '.join(result_types)}")
    print(f"Output ZIP         : {ZIP_FILE}")
    print(f"ZIP size           : {ZIP_FILE.stat().st_size / 1024**2:.2f} MB")
    print("=" * 60)


# ============================================================
# UNPACK
# ============================================================

def unpack(output_dir, all_mode=False):

    result_types = get_result_types(all_mode)
    destination = BASE / output_dir

    if not ZIP_FILE.is_file():
        raise FileNotFoundError(f"ZIP not found: {ZIP_FILE}")

    destination.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UNPACK")
    print("=" * 60)
    print(f"ZIP         : {ZIP_FILE}")
    print(f"Destination : {destination}")
    print(f"Types       : {', '.join(result_types)}")
    print()

    extracted = 0
    skipped = 0

    with zipfile.ZipFile(ZIP_FILE, "r") as archive:

        for info in archive.infolist():

            if info.is_dir():
                continue

            relative = valid_archive_path(
                info.filename,
                result_types
            )

            if relative is None:
                skipped += 1
                continue

            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)

            with archive.open(info, "r") as src:
                with open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            extracted += 1

            if extracted % 1000 == 0:
                print(f"  Extracted: {extracted} files")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("UNPACK COMPLETE")
    print("=" * 60)
    print(f"Files extracted : {extracted}")
    print(f"Files skipped   : {skipped}")
    print(f"Destination     : {destination}")
    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Merge and synchronize Fail2Drive results."
    )

    parser.add_argument(
        "action",
        choices=("collect", "unpack"),
        help="collect: merge ZIP + local files; unpack: extract ZIP"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Include res/, err/, out/, viz/"
    )

    parser.add_argument(
        "--out-dir",
        choices=("results", "results_shome"),
        default="results",
        help="Local source/destination directory (default: results)"
    )

    args = parser.parse_args()

    if args.action == "collect":
        collect(args.out_dir, args.all)

    else:
        unpack(args.out_dir, args.all)


if __name__ == "__main__":
    main()
