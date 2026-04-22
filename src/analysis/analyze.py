import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run reference and similarity analysis in parallel")
    ap.add_argument("--in-dir", required=True,
                    help="Path to the verified_json directory")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for analysis results")
    ap.add_argument("--show", action="store_true", help="Show plots interactively")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--fig-width", type=float, default=12)
    ap.add_argument("--support-thresh", type=float, default=0.62)
    ap.add_argument("--related-thresh", type=float, default=0.42)
    ap.add_argument("--faculty-name", default="default", help="Label to use instead of 'default' when in-dir has no subdirectories")
    return ap.parse_args()


def run_script(cmd: list[str], label: str) -> tuple[str, int, str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    if result.stderr:
        output += "\n" + result.stderr
    return label, result.returncode, output


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    refs_out = out_dir / "refs"
    sim_out = out_dir / "similarity"

    if not in_dir.exists():
        print(f"Error: --in-dir does not exist: {in_dir}", file=sys.stderr)
        sys.exit(1)

    refs_cmd = [
        sys.executable, str(SCRIPT_DIR / "analyze_refs.py"),
        "--in-dir", str(in_dir),
        "--out-dir", str(refs_out),
        "--dpi", str(args.dpi),
        "--fig-width", str(args.fig_width),
    ]
    if args.show:
        refs_cmd.append("--show")
    if args.faculty_name:
        refs_cmd += ["--faculty-name", args.faculty_name]

    sim_cmd = [
        sys.executable, str(SCRIPT_DIR / "analyze_similarity.py"),
        "--in-dir", str(in_dir),
        "--out-dir", str(sim_out),
        "--dpi", str(args.dpi),
        "--fig-width", str(args.fig_width),
        "--support-thresh", str(args.support_thresh),
        "--related-thresh", str(args.related_thresh),
    ]
    if args.show:
        sim_cmd.append("--show")
    if args.faculty_name:
        sim_cmd += ["--faculty-name", args.faculty_name]

    tasks = [
        (refs_cmd, "analyze_refs"),
        (sim_cmd, "analyze_similarity"),
    ]

    if args.show:
        # run serially so plt.show() works
        all_ok = True
        for cmd, label in tasks:
            print(f"\n[analyze] Running {label} ...")
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                print(f"[analyze] {label} failed (exit {rc})", file=sys.stderr)
                all_ok = False
        sys.exit(0 if all_ok else 1)
    else:
        print("[analyze] Running analyze_refs and analyze_similarity in parallel ...\n")
        results = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(run_script, cmd, label): label
                       for cmd, label in tasks}
            for future in as_completed(futures):
                label, rc, output = future.result()
                results[label] = (rc, output)
                # Stream output as it arrives
                prefix = f"[{label}]"
                for line in output.splitlines():
                    print(f"{prefix} {line}")

        print("\n[analyze] Summary:")
        all_ok = True
        for label, (rc, _) in results.items():
            status = "OK" if rc == 0 else f"FAILED (exit {rc})"
            print(f"  {label}: {status}")
            if rc != 0:
                all_ok = False

        if all_ok:
            print(f"\n[analyze] Done. Results in:")
            print(f"  {refs_out}")
            print(f"  {sim_out}")
        sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
