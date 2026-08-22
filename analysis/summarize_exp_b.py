"""Reconstruct Experiment B results from individual run artifacts.

The archived ``aggregate.csv`` files are intentionally ignored: parallel seed
processes overwrote that shared path during the Kaggle runs.  This script reads
only per-cell ``summary.json`` and ``history.json`` files, either from normal
directories or directly from ZIP archives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


METHODS = ("imp", "one_shot")
METHOD_LABELS = {"imp": "Stateful IMP", "one_shot": "Post-grok one-shot"}
SPARSITIES = (0.0, 0.2, 0.5, 0.7, 0.8, 0.9, 0.95)
SEEDS = (0, 1, 2, 3, 4)

_CELL_RE = re.compile(
    r"(?:^|[!/])exp_b/(imp|one_shot)/sp_([0-9.]+)/seed_([0-9]+)/"
    r"grok_phase/(summary|history)\.json$"
)
_PRETRAIN_RE = re.compile(
    r"(?:^|[!/])exp_b/one_shot/sp_([0-9.]+)/seed_([0-9]+)/"
    r"oneshot_pretrain/(summary|history)\.json$"
)
_IMP_RE = re.compile(
    r"(?:^|[!/])exp_b/imp/sp_([0-9.]+)/seed_([0-9]+)/"
    r"imp_phase/(summary|history)\.json$"
)


class ResultValidationError(RuntimeError):
    """Raised when the result archive set is ambiguous or incomplete."""


@dataclass(frozen=True)
class Artifact:
    source: str
    data: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def json(self) -> dict[str, Any]:
        try:
            value = json.loads(self.data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResultValidationError(f"invalid JSON in {self.source}: {exc}") from exc
        if not isinstance(value, dict):
            raise ResultValidationError(f"expected a JSON object in {self.source}")
        return value


@dataclass
class ExpBData:
    cells: list[dict[str, Any]]
    pretrain_cells: list[dict[str, Any]]
    pretrain_by_seed: list[dict[str, Any]]
    imp_discovery: list[dict[str, Any]]


def _sp(value: float | str) -> float:
    return round(float(value), 8)


def _event_step(value: Any) -> int | None:
    if value is None:
        return None
    step = int(value)
    return step if step >= 0 else None


def _read_artifacts(inputs: Sequence[Path]) -> list[Artifact]:
    artifacts: list[Artifact] = []
    visited_zips: set[Path] = set()
    visited_files: set[Path] = set()
    workspace = Path.cwd().resolve()

    def source_label(path: Path) -> str:
        try:
            return path.resolve().relative_to(workspace).as_posix()
        except ValueError:
            return str(path.resolve())

    def add_file(path: Path) -> None:
        resolved = path.resolve()
        if resolved in visited_files:
            return
        visited_files.add(resolved)
        normalized = path.as_posix()
        if _CELL_RE.search(normalized) or _PRETRAIN_RE.search(normalized) or _IMP_RE.search(normalized):
            artifacts.append(Artifact(source_label(resolved), path.read_bytes()))

    def add_zip(path: Path) -> None:
        resolved = path.resolve()
        if resolved in visited_zips:
            return
        visited_zips.add(resolved)
        try:
            with zipfile.ZipFile(path) as archive:
                for name in sorted(archive.namelist()):
                    normalized = name.replace("\\", "/")
                    if not (
                        _CELL_RE.search(normalized)
                        or _PRETRAIN_RE.search(normalized)
                        or _IMP_RE.search(normalized)
                    ):
                        continue
                    artifacts.append(
                        Artifact(f"{source_label(resolved)}!{normalized}", archive.read(name))
                    )
        except (OSError, zipfile.BadZipFile) as exc:
            raise ResultValidationError(f"cannot read ZIP archive {resolved}: {exc}") from exc

    for raw in inputs:
        path = raw.resolve()
        if not path.exists():
            raise ResultValidationError(f"input does not exist: {path}")
        if path.is_file():
            add_zip(path) if zipfile.is_zipfile(path) else add_file(path)
            continue
        for zip_path in sorted(path.rglob("*.zip")):
            add_zip(zip_path)
        for json_path in sorted(path.rglob("*.json")):
            add_file(json_path)
    return artifacts


def _insert_unique(
    records: dict[tuple[Any, ...], Artifact], key: tuple[Any, ...], artifact: Artifact
) -> None:
    previous = records.get(key)
    if previous is not None:
        identity = "identical" if previous.sha256 == artifact.sha256 else "conflicting"
        raise ResultValidationError(
            f"duplicate {identity} artifact for {key}: {previous.source} and {artifact.source}"
        )
    records[key] = artifact


def _validate_expected(
    actual: Iterable[tuple[Any, ...]], expected: Iterable[tuple[Any, ...]], label: str
) -> None:
    actual_set, expected_set = set(actual), set(expected)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        raise ResultValidationError(f"{label}: missing={missing}, extra={extra}")


def _parse_cell(
    key: tuple[str, float, int], summary_artifact: Artifact, history_artifact: Artifact
) -> dict[str, Any]:
    method, sparsity, seed = key
    summary = summary_artifact.json()
    history = history_artifact.json()
    config = summary.get("config") or {}
    if not isinstance(config, dict):
        raise ResultValidationError(f"invalid config object in {summary_artifact.source}")

    claimed = (config.get("method"), _sp(config.get("target_sparsity", -1)), config.get("seed"))
    if claimed != key:
        raise ResultValidationError(
            f"path/config mismatch in {summary_artifact.source}: path={key}, config={claimed}"
        )

    mem = _event_step(summary.get("memorization_step"))
    grok = _event_step(summary.get("grokking_step"))
    grokked = bool(summary.get("grokked"))
    if grokked != (grok is not None):
        raise ResultValidationError(f"grokked/event mismatch in {summary_artifact.source}")
    expected_gap = grok - mem if grok is not None and mem is not None else None
    raw_gap = summary.get("grokking_gap")
    normalized_raw_gap = (
        int(raw_gap) if expected_gap is not None and raw_gap is not None
        else None if raw_gap is None or int(raw_gap) == -1
        else int(raw_gap)
    )
    if normalized_raw_gap != expected_gap:
        raise ResultValidationError(
            f"grokking-gap mismatch in {summary_artifact.source}: "
            f"{normalized_raw_gap} != {expected_gap}"
        )
    gap = expected_gap

    steps = [int(step) for step in history.get("steps", [])]
    val_acc = [float(value) for value in history.get("val_acc", [])]
    if not steps or len(steps) != len(val_acc):
        raise ResultValidationError(f"invalid step/validation history in {history_artifact.source}")

    run, live_index = 0, None
    for index, value in enumerate(val_acc):
        run = run + 1 if value >= 0.95 else 0
        if run == 2:
            live_index = index
            break
    live_confirmation = steps[live_index] if live_index is not None else None
    evals_from_confirmation = len(steps) - live_index if live_index is not None else None

    actual_sparsity = float(summary.get("actual_sparsity"))
    if not math.isfinite(actual_sparsity) or not 0.0 <= actual_sparsity < 1.0:
        raise ResultValidationError(f"invalid actual sparsity in {summary_artifact.source}")

    return {
        "method": method,
        "target_sparsity": sparsity,
        "actual_sparsity": actual_sparsity,
        "seed": seed,
        "memorized": mem is not None,
        "grokked": grokked,
        "memorization_step": mem,
        "grokking_step": grok,
        "grokking_gap": gap,
        "final_train_acc": float(summary.get("final_train_acc")),
        "final_val_acc": float(summary.get("final_val_acc")),
        "last_logged_step": steps[-1],
        "live_confirmation_step": live_confirmation,
        "evals_from_live_confirmation": evals_from_confirmation,
        "stopped_before_budget": steps[-1] < int(config.get("n_grok_steps", 40_000)),
        "weight_decay": float(config.get("weight_decay")),
        "n_grok_steps": int(config.get("n_grok_steps")),
        "summary_source": summary_artifact.source,
        "history_source": history_artifact.source,
    }


def _parse_phase_summary(
    artifact: Artifact, method: str, sparsity: float, seed: int
) -> dict[str, Any]:
    summary = artifact.json()
    mem = _event_step(summary.get("memorization_step"))
    grok = _event_step(summary.get("grokking_step"))
    expected_gap = grok - mem if grok is not None and mem is not None else None
    raw_gap = summary.get("grokking_gap")
    normalized_raw_gap = (
        int(raw_gap) if expected_gap is not None and raw_gap is not None
        else None if raw_gap is None or int(raw_gap) == -1
        else int(raw_gap)
    )
    if normalized_raw_gap != expected_gap:
        raise ResultValidationError(f"phase gap mismatch in {artifact.source}")
    gap = expected_gap
    return {
        "method": method,
        "target_sparsity": sparsity,
        "seed": seed,
        "memorized": mem is not None,
        "grokked": bool(summary.get("grokked")),
        "memorization_step": mem,
        "grokking_step": grok,
        "grokking_gap": gap,
        "final_train_acc": float(summary.get("final_train_acc")),
        "final_val_acc": float(summary.get("final_val_acc")),
        "phase_input_sparsity": float(summary.get("actual_sparsity", 0.0)),
        "summary_sha256": artifact.sha256,
        "summary_source": artifact.source,
    }


def load_exp_b(inputs: Sequence[Path], strict: bool = True) -> ExpBData:
    """Load, validate, and normalize the Exp B result archive set."""
    cell_summaries: dict[tuple[str, float, int], Artifact] = {}
    cell_histories: dict[tuple[str, float, int], Artifact] = {}
    pre_summaries: dict[tuple[float, int], Artifact] = {}
    pre_histories: dict[tuple[float, int], Artifact] = {}
    imp_summaries: dict[tuple[float, int], Artifact] = {}
    imp_histories: dict[tuple[float, int], Artifact] = {}

    for artifact in _read_artifacts(inputs):
        normalized = artifact.source.replace("\\", "/")
        match = _CELL_RE.search(normalized)
        if match:
            key = (match[1], _sp(match[2]), int(match[3]))
            target = cell_summaries if match[4] == "summary" else cell_histories
            _insert_unique(target, key, artifact)
            continue
        match = _PRETRAIN_RE.search(normalized)
        if match:
            key = (_sp(match[1]), int(match[2]))
            target = pre_summaries if match[3] == "summary" else pre_histories
            _insert_unique(target, key, artifact)
            continue
        match = _IMP_RE.search(normalized)
        if match:
            key = (_sp(match[1]), int(match[2]))
            target = imp_summaries if match[3] == "summary" else imp_histories
            _insert_unique(target, key, artifact)

    if strict:
        expected_cells = [(method, sp, seed) for method in METHODS for sp in SPARSITIES for seed in SEEDS]
        _validate_expected(cell_summaries, expected_cells, "grok summaries")
        _validate_expected(cell_histories, expected_cells, "grok histories")
        expected_sparse = [(sp, seed) for sp in SPARSITIES if sp > 0 for seed in SEEDS]
        _validate_expected(pre_summaries, expected_sparse, "one-shot pretrain summaries")
        _validate_expected(pre_histories, expected_sparse, "one-shot pretrain histories")
        _validate_expected(imp_summaries, expected_sparse, "IMP phase summaries")
        _validate_expected(imp_histories, expected_sparse, "IMP phase histories")

    if set(cell_summaries) != set(cell_histories):
        raise ResultValidationError("grok summary/history key sets differ")
    cells = [
        _parse_cell(key, cell_summaries[key], cell_histories[key])
        for key in sorted(cell_summaries)
    ]

    if strict:
        for seed in SEEDS:
            dense = [row for row in cells if row["seed"] == seed and row["target_sparsity"] == 0]
            comparable = (
                "memorization_step", "grokking_step", "grokking_gap", "final_train_acc",
                "final_val_acc", "last_logged_step",
            )
            if len(dense) != 2 or any(dense[0][field] != dense[1][field] for field in comparable):
                raise ResultValidationError(f"dense method copies differ for seed {seed}")

    pretrain_cells = [
        _parse_phase_summary(artifact, "one_shot", key[0], key[1])
        for key, artifact in sorted(pre_summaries.items())
    ]
    pretrain_by_seed: list[dict[str, Any]] = []
    for seed in sorted({row["seed"] for row in pretrain_cells}):
        rows = [row for row in pretrain_cells if row["seed"] == seed]
        hashes = {row["summary_sha256"] for row in rows}
        history_hashes = {pre_histories[(row["target_sparsity"], seed)].sha256 for row in rows}
        identical = len(hashes) == 1 and len(history_hashes) == 1
        if strict and not identical:
            raise ResultValidationError(f"one-shot warm-up differs across sparsities for seed {seed}")
        first = rows[0]
        pretrain_by_seed.append({
            "seed": seed,
            "sparsity_count": len(rows),
            "sparsities": ";".join(f"{row['target_sparsity']:.2f}" for row in rows),
            "identical_across_sparsities": identical,
            "memorized": first["memorized"],
            "grokked": first["grokked"],
            "memorization_step": first["memorization_step"],
            "grokking_step": first["grokking_step"],
            "grokking_gap": first["grokking_gap"],
            "final_train_acc": first["final_train_acc"],
            "final_val_acc": first["final_val_acc"],
            "summary_sha256": first["summary_sha256"],
        })

    imp_discovery: list[dict[str, Any]] = []
    for (sparsity, seed), artifact in sorted(imp_summaries.items()):
        row = _parse_phase_summary(artifact, "imp", sparsity, seed)
        n_rounds = math.ceil(math.log(1.0 - sparsity) / math.log(0.8))
        history = imp_histories[(sparsity, seed)].json()
        steps = [int(step) for step in history.get("steps", [])]
        train_acc = [float(value) for value in history.get("train_acc", [])]
        val_acc = [float(value) for value in history.get("val_acc", [])]
        row.update({
            "n_rounds": n_rounds,
            "steps_per_round": 400,
            "total_discovery_updates": n_rounds * 400,
            "preserved_round": n_rounds,
            "preserved_history_last_step": steps[-1] if steps else None,
            "preserved_round_max_train_acc": max(train_acc) if train_acc else None,
            "preserved_round_max_val_acc": max(val_acc) if val_acc else None,
            "preserved_round_any_val_at_threshold": any(value >= 0.95 for value in val_acc),
            "intermediate_rounds_preserved": False,
        })
        imp_discovery.append(row)

    return ExpBData(cells, pretrain_cells, pretrain_by_seed, imp_discovery)


def _numeric_stats(
    values: Sequence[int | float | None], prefix: str, *, allow_negative: bool = False
) -> dict[str, Any]:
    valid = [
        float(value) for value in values
        if value is not None and (allow_negative or float(value) >= 0)
    ]
    result: dict[str, Any] = {f"{prefix}_n": len(valid)}
    for name in ("mean", "median", "std", "min", "max"):
        result[f"{prefix}_{name}"] = None
    if not valid:
        return result
    result[f"{prefix}_mean"] = statistics.fmean(valid)
    result[f"{prefix}_median"] = statistics.median(valid)
    result[f"{prefix}_std"] = statistics.stdev(valid) if len(valid) > 1 else None
    result[f"{prefix}_min"] = min(valid)
    result[f"{prefix}_max"] = max(valid)
    return result


def summarize_cells(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return one statistically explicit row per method/sparsity condition."""
    output: list[dict[str, Any]] = []
    groups = sorted({(str(row["method"]), float(row["target_sparsity"])) for row in cells})
    for method, sparsity in groups:
        group = [
            row for row in cells
            if row["method"] == method and float(row["target_sparsity"]) == sparsity
        ]
        n_grokked = sum(bool(row["grokked"]) for row in group)
        n_memorized = sum(bool(row["memorized"]) for row in group)
        result: dict[str, Any] = {
            "method": method,
            "target_sparsity": sparsity,
            "n_runs": len(group),
            "n_memorized": n_memorized,
            "n_grokked": n_grokked,
            "success_rate": n_grokked / len(group) if group else None,
            "n_stopped_before_budget": sum(bool(row["stopped_before_budget"]) for row in group),
        }
        result.update(_numeric_stats([row["memorization_step"] for row in group], "memorization_step"))
        result.update(_numeric_stats([row["grokking_step"] for row in group], "grokking_step"))
        result.update(_numeric_stats(
            [row["grokking_gap"] for row in group], "grokking_gap", allow_negative=True
        ))
        result.update(_numeric_stats([row["final_val_acc"] for row in group], "final_val_acc"))
        output.append(result)
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ResultValidationError(f"refusing to write empty table: {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in columns})


def _markdown(summary: Sequence[Mapping[str, Any]], pretrain: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Experiment B results",
        "",
        "Generated from individual `grok_phase/summary.json` and `history.json` artifacts.",
        "Archived `aggregate.csv` files were ignored because parallel workers overwrote them.",
        "",
        "## Final grokking success",
        "",
        "| Method | Sparsity | Success | Rate | Median grok step (successes only) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        median = row["grokking_step_median"]
        median_text = f"{float(median):g}" if median is not None else "NA"
        lines.append(
            f"| {METHOD_LABELS[row['method']]} | {float(row['target_sparsity']):.0%} | "
            f"{row['n_grokked']}/{row['n_runs']} | {float(row['success_rate']):.0%} | "
            f"{median_text} |"
        )
    lines.extend([
        "",
        "`NA` means that no seed met the two-evaluation 95% validation threshold within 40,000 steps.",
        "Failure sentinels (`-1`) are represented as missing and are never averaged.",
        "",
        "## Dense warm-up used for one-shot masks",
        "",
        "| Seed | Repeated sparsities | Identical | Memorization step | Grokking step | Final val accuracy |",
        "|---:|---:|:---:|---:|---:|---:|",
    ])
    for row in pretrain:
        lines.append(
            f"| {row['seed']} | {row['sparsity_count']} | {row['identical_across_sparsities']} | "
            f"{row['memorization_step']} | {row['grokking_step']} | "
            f"{float(row['final_val_acc']):.6f} |"
        )
    lines.extend([
        "",
        "All warm-ups grokked before magnitude measurement at update 1,200. The one-shot masks",
        "were selected after grokking and rewound to initialization.",
        "",
    ])
    return "\n".join(lines)


def _plot_success(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"imp": "#217a4b", "one_shot": "#d06b27"}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method in METHODS:
        rows = [row for row in summary if row["method"] == method]
        x = [100 * float(row["target_sparsity"]) for row in rows]
        y = [100 * float(row["success_rate"]) for row in rows]
        ax.plot(x, y, marker="o", linewidth=2, color=colors[method], label=METHOD_LABELS[method])
        annotation_y = 8 if method == "imp" else -14
        for px, py, row in zip(x, y, rows):
            ax.annotate(
                f"{row['n_grokked']}/{row['n_runs']}", (px, py), xytext=(0, annotation_y),
                textcoords="offset points", ha="center", fontsize=8,
            )
    ax.set(xlabel="Target sparsity (%)", ylabel="Grokking success rate (%)", ylim=(-5, 108))
    ax.set_xticks([0, 20, 50, 70, 80, 90, 95])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_timings(cells: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"imp": "#217a4b", "one_shot": "#d06b27"}
    offsets = {"imp": -1.2, "one_shot": 1.2}
    fig, (ax, failures) = plt.subplots(
        2, 1, figsize=(8.1, 5.8), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
    )
    for method in METHODS:
        rows = [row for row in cells if row["method"] == method]
        successful = [row for row in rows if row["grokking_step"] is not None]
        ax.scatter(
            [100 * float(row["target_sparsity"]) + offsets[method] for row in successful],
            [row["grokking_step"] for row in successful],
            color=colors[method], alpha=0.82, s=30, label=METHOD_LABELS[method],
        )
        failed = [row for row in rows if row["grokking_step"] is None]
        failures.scatter(
            [
                100 * float(row["target_sparsity"]) + offsets[method]
                + (int(row["seed"]) - 2) * 0.32
                for row in failed
            ],
            [0 if method == "imp" else 1 for _ in failed],
            color=colors[method], marker="x", s=38, linewidths=1.6,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Reported grokking step (log scale)")
    ax.grid(axis="y", which="both", alpha=0.22)
    ax.legend(frameon=False, fontsize=9)
    failures.set_yticks([0, 1], ["Stateful IMP DNF", "Post-grok one-shot DNF"])
    failures.set_ylim(-0.7, 1.7)
    failures.grid(axis="x", alpha=0.2)
    failures.set_xlabel("Target sparsity (%)")
    failures.set_xticks([0, 20, 50, 70, 80, 90, 95])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_outputs(data: ExpBData, output_dir: Path, figures: bool = False) -> None:
    summary = summarize_cells(data.cells)
    _write_csv(output_dir / "exp_b_cells.csv", data.cells)
    _write_csv(output_dir / "exp_b_summary.csv", summary)
    _write_csv(output_dir / "oneshot_pretrain_cells.csv", data.pretrain_cells)
    _write_csv(output_dir / "oneshot_pretrain_by_seed.csv", data.pretrain_by_seed)
    _write_csv(output_dir / "imp_discovery_last_preserved_round.csv", data.imp_discovery)
    (output_dir / "exp_b_results.md").write_text(_markdown(summary, data.pretrain_by_seed), encoding="utf-8")
    if figures:
        _plot_success(summary, output_dir / "exp_b_success_rate.png")
        _plot_timings(data.cells, output_dir / "exp_b_grokking_steps.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=[Path("res/raw")])
    parser.add_argument("--output-dir", type=Path, default=Path("res/derived"))
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()

    data = load_exp_b(args.inputs, strict=not args.allow_incomplete)
    write_outputs(data, args.output_dir, figures=args.figures)
    print(
        f"validated {len(data.cells)} final cells, {len(data.pretrain_cells)} one-shot warm-ups, "
        f"and {len(data.imp_discovery)} last-preserved IMP phases"
    )
    print(f"derived outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
