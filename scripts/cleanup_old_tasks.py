"""
Prune old pipeline output: uploads/, meshes/, tasks/, cache/.

Nothing here expires on its own. uploads/ and meshes/ were measured at 4.6GB
and 8.3GB respectively partway through this session with zero eviction of any
kind — every upload, including failed/duplicate/corrupt ones, permanently
consumes disk. A manual script, not a background job inside the app: this is a
local/demo deployment, not a production service, so an in-process scheduler
would be more risk (silently deleting something mid-use) than it's worth.
Run it periodically by hand, or wire it into cron/Task Scheduler yourself if
you want it automatic.

Age is judged by each entry's own mtime — for uploads/*, the file's mtime; for
meshes/{task_id}/ and tasks/{task_id}/, the directory's mtime (updates when a
file is added/removed directly inside it, which covers both the initial
pipeline write and a later /api/resect call touching the same directory).

Defaults to a dry run: prints what WOULD be removed and its total size, deletes
nothing, unless --delete is passed. This is deliberate — the four directories
below hold everything a task needs to be viewable again, so a wrong --days
value should be caught by reading the dry-run output, not discovered after the
fact.

Usage:
    python scripts/cleanup_old_tasks.py --days 30           # dry run (default)
    python scripts/cleanup_old_tasks.py --days 30 --delete  # actually remove
"""
import argparse
import shutil
import time
from pathlib import Path

TARGETS = {
    "uploads": {"dir": Path("uploads"), "kind": "files"},
    "meshes": {"dir": Path("meshes"), "kind": "dirs"},
    "tasks": {"dir": Path("tasks"), "kind": "dirs"},
    "cache": {"dir": Path("cache"), "kind": "files"},
}


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


def find_stale_entries(base_dir: Path, kind: str, cutoff_ts: float) -> list[Path]:
    if not base_dir.exists():
        return []
    entries = base_dir.iterdir()
    if kind == "files":
        candidates = (p for p in entries if p.is_file())
    else:
        candidates = (p for p in entries if p.is_dir())
    return [p for p in candidates if p.stat().st_mtime < cutoff_ts]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=float, default=30,
                        help="Remove entries older than this many days (default: 30)")
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete. Without this flag, only reports what would happen.")
    parser.add_argument("--only", choices=list(TARGETS), default=None,
                        help="Limit to a single target directory, e.g. --only uploads")
    args = parser.parse_args()

    cutoff_ts = time.time() - args.days * 86400
    targets = {args.only: TARGETS[args.only]} if args.only else TARGETS

    print(f"{'DRY RUN — nothing will be deleted' if not args.delete else 'DELETING'}: "
          f"entries older than {args.days:g} days\n")

    grand_total_bytes = 0
    grand_total_count = 0

    for name, spec in targets.items():
        stale = find_stale_entries(spec["dir"], spec["kind"], cutoff_ts)
        if not stale:
            print(f"{name}: nothing to remove")
            continue

        total_bytes = sum(_size(p) for p in stale)
        grand_total_bytes += total_bytes
        grand_total_count += len(stale)
        print(f"{name}: {len(stale)} entries, {_human(total_bytes)}")

        for p in stale:
            age_days = (time.time() - p.stat().st_mtime) / 86400
            print(f"  {'DELETE' if args.delete else '  would delete'} "
                  f"{p.name}  ({age_days:.0f}d old, {_human(_size(p))})")
            if args.delete:
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p)

    print(f"\nTotal: {grand_total_count} entries, {_human(grand_total_bytes)}"
          f"{'' if args.delete else ' (dry run — re-run with --delete to actually remove)'}")


if __name__ == "__main__":
    main()
