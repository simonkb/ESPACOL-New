#!/usr/bin/env python3
"""
diff_configs.py — diff two `Config: <Name>(...)` lines from training logs.

Purpose
-------
The 84.25% multi-tile run and the 79.28% full-ESPAOCL run must be proven to
differ ONLY by the flag(s) under test before any accuracy gap can be attributed
to the concept spine. This prints exactly which config fields differ, including
fields present in one run but absent in the other (schema drift between config
generations is itself a red flag — it means the runs aren't comparable).

Usage
-----
    # from files (searches each file for its first `Config:` line)
    python diff_configs.py runA.log runB.log

    # or paste the two `DRConfig(...)` strings directly as args
    python diff_configs.py "DRConfig(img_size=300, ...)" "DRConfig(img_size=300, ...)"

    # label the columns (default: A / B)
    python diff_configs.py runA.log runB.log --labels 84.25 79.28

Notes
-----
- Parsing is regex-based on `key=value` pairs. It handles ints, floats, bools,
  quoted strings, and None. It does NOT handle values containing a top-level
  comma (e.g. an inline tuple/list) — none of the ESPAOCL configs have those,
  but the script warns if it sees an unbalanced paren so you don't trust a bad parse.
"""

from __future__ import annotations

import argparse
import os
import re
import sys


CONFIG_RE = re.compile(r"Config:\s*(\w+)\((.*)\)\s*$")
# Fallback: any `<Name>( ... )` if the line has no "Config:" prefix (pasted arg).
BARE_RE = re.compile(r"^\s*(\w+)\((.*)\)\s*$")
KV_RE = re.compile(r"(\w+)=(.*?)(?=,\s*\w+=|$)")


def extract_config(source: str) -> tuple[str, str]:
    """Return (config_class_name, inner_kv_string) from a file path or literal string."""
    text = None
    if os.path.isfile(source):
        with open(source, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "Config:" in line and "(" in line:
                    text = line.rstrip("\n")
                    break
        if text is None:
            sys.exit(f"ERROR: no 'Config: <Name>(...)' line found in {source}")
    else:
        text = source.strip()

    m = CONFIG_RE.search(text) or BARE_RE.search(text)
    if not m:
        sys.exit(
            f"ERROR: could not parse a config from:\n  {text[:200]}\n"
            "Expected a line like  Config: DRConfig(key=value, ...)"
        )
    name, inner = m.group(1), m.group(2)
    if inner.count("(") != inner.count(")"):
        print(
            f"WARNING: unbalanced parentheses in {name}(...) — a value may contain "
            "a comma and the parse below could be wrong.",
            file=sys.stderr,
        )
    return name, inner


def parse_kv(inner: str) -> dict[str, str]:
    d: dict[str, str] = {}
    for m in KV_RE.finditer(inner):
        d[m.group(1)] = m.group(2).strip()
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description="Diff two training-log Config lines.")
    ap.add_argument("a", help="log file path or literal Config string (run A)")
    ap.add_argument("b", help="log file path or literal Config string (run B)")
    ap.add_argument(
        "--labels",
        nargs=2,
        metavar=("A", "B"),
        default=["A", "B"],
        help="column labels (e.g. --labels 84.25 79.28)",
    )
    args = ap.parse_args()

    name_a, inner_a = extract_config(args.a)
    name_b, inner_b = extract_config(args.b)
    cfg_a, cfg_b = parse_kv(inner_a), parse_kv(inner_b)
    la, lb = args.labels

    print(f"\n  {la}: {name_a}   ({len(cfg_a)} fields)")
    print(f"  {lb}: {name_b}   ({len(cfg_b)} fields)")
    if name_a != name_b:
        print(f"  !! different config classes ({name_a} vs {name_b}) — schemas may not align")
    print()

    all_keys = sorted(set(cfg_a) | set(cfg_b))
    only_a = [k for k in all_keys if k not in cfg_b]
    only_b = [k for k in all_keys if k not in cfg_a]
    differing = [k for k in all_keys if k in cfg_a and k in cfg_b and cfg_a[k] != cfg_b[k]]

    kw = max([len(k) for k in all_keys] + [5])
    vw = max([len(v) for v in list(cfg_a.values()) + list(cfg_b.values())] + [len(la), len(lb), 4])

    def row(k: str, va: str, vb: str) -> str:
        return f"  {k:<{kw}}  {va:<{vw}}  {vb:<{vw}}"

    if differing:
        print("DIFFERING FIELDS (present in both, values differ):")
        print(row("field", la, lb))
        print("  " + "-" * (kw + 2 * vw + 4))
        for k in differing:
            print(row(k, cfg_a[k], cfg_b[k]))
    else:
        print("No differing values among shared fields.")

    if only_a:
        print(f"\nONLY IN {la} (missing from {lb} -- schema drift):")
        for k in only_a:
            print(f"  {k} = {cfg_a[k]}")
    if only_b:
        print(f"\nONLY IN {lb} (missing from {la} -- schema drift):")
        for k in only_b:
            print(f"  {k} = {cfg_b[k]}")

    # Focused verdict on the suspects that matter for the spine attribution.
    print("\nSUSPECT CHECK (the fields that decide whether the gap is the spine):")
    suspects = [
        "use_concept_spine",
        "use_image_text",
        "gamma",
        "finetune_text_encoder",
        "text_finetune_start_epoch",
        "text_finetune_layers",
        "use_multi_tile",
        "epochs",
        "early_stop_patience",
        "lr",
        "lr_patience",
        "eta",
        "delta",
        "nu",
    ]
    for k in suspects:
        va, vb = cfg_a.get(k, "<absent>"), cfg_b.get(k, "<absent>")
        flag = "" if va == vb else "   <-- DIFFERS"
        print(f"  {k:<28} {la}={va:<12} {lb}={vb}{flag}")

    n_real_diffs = len(differing) + len(only_a) + len(only_b)
    print(
        f"\nVERDICT: {n_real_diffs} field(s) differ. "
        + (
            "If any is NOT `use_concept_spine`, the runs are UNMATCHED and no gap "
            "can be attributed to the spine yet."
            if n_real_diffs
            else "Configs are identical — this can't be your two runs; re-check the files."
        )
    )


if __name__ == "__main__":
    main()
