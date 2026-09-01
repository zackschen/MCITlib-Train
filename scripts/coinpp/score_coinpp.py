#!/usr/bin/env python3
"""Score CoIN++ prediction matrices and summarize continual-learning behavior."""

import argparse
import csv
import json
import re
import string
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FACTORS = ("visual_substrate", "skill_requirement", "evidence_complexity")
ANLS_DATASETS = {"docvqa", "infographicvqa"}
RELAXED_NUMERIC_DATASETS = {"chartqa", "chartqa_eval", "dvqa"}
ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)")
STAGE_RE = re.compile(r"^(?:stage)?0*(\d+)_")
PUNCT_TRANSLATION = str.maketrans({char: " " for char in string.punctuation})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--factor", choices=FACTORS, required=True)
    parser.add_argument("--profile", choices=("clean_5k", "strict", "legacy_10k"), default="strict")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "CoIN++",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=REPO_ROOT / "results" / "CoIN++",
    )
    parser.add_argument("--eval-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Malformed JSONL {}:{}: {}".format(path, line_number, error)
                )
    return rows


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def normalize_text(value):
    text = str(value or "").lower().strip()
    text = text.replace("\n", " ")
    text = text.translate(PUNCT_TRANSLATION)
    text = ARTICLES.sub(" ", text)
    return " ".join(text.split())


def references_for_row(row):
    values = row.get("answers")
    if not values:
        values = [row.get("gt")]
    references = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            value = (
                value.get("answer")
                or value.get("text")
                or value.get("value")
                or value.get("label")
            )
        if value is not None and str(value).strip():
            references.append(str(value).strip())
    return references


def exact_score(prediction, references):
    normalized = normalize_text(prediction)
    return float(bool(normalized) and any(normalized == normalize_text(ref) for ref in references))


def token_f1_single(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens and bool(ref_tokens))
    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common = sum(overlap.values())
    if not common:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def token_f1_score(prediction, references):
    return max(
        (token_f1_single(prediction, reference) for reference in references),
        default=0.0,
    )


def levenshtein(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def anls_single(prediction, reference):
    prediction = normalize_text(prediction)
    reference = normalize_text(reference)
    if not prediction or not reference:
        return float(prediction == reference and bool(reference))
    similarity = 1.0 - levenshtein(prediction, reference) / max(
        len(prediction), len(reference)
    )
    return similarity if similarity >= 0.5 else 0.0


def anls_score(prediction, references):
    return max(
        (anls_single(prediction, reference) for reference in references),
        default=0.0,
    )


def numeric_values(value):
    output = []
    for match in NUMBER_RE.findall(str(value or "")):
        try:
            output.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return output


def numeric_relaxed_score(prediction, references):
    if exact_score(prediction, references):
        return 1.0
    predicted = numeric_values(prediction)
    for reference in references:
        targets = numeric_values(reference)
        for pred_value in predicted:
            for target in targets:
                tolerance = 0.05 * abs(target) if target else 0.05
                if abs(pred_value - target) <= tolerance + 1e-12:
                    return 1.0
    return 0.0


def normalize_choices(value):
    if isinstance(value, dict):
        return [str(item) for _, item in sorted(value.items(), key=lambda pair: str(pair[0]))]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def option_index(value, choices):
    normalized = normalize_text(value)
    if not normalized:
        return None
    for index, choice in enumerate(choices):
        if normalized == normalize_text(choice):
            return index
    match = re.match(r"^\s*[\(\[]?([a-z])[\)\].:\s]*$", str(value or ""), re.I)
    if match:
        index = ord(match.group(1).lower()) - ord("a")
        return index if 0 <= index < len(choices) else None
    match = re.match(r"^\s*[\(\[]?(\d+)[\)\].:\s]*$", str(value or ""))
    if match:
        number = int(match.group(1))
        if 0 <= number < len(choices):
            return number
        if 1 <= number <= len(choices):
            return number - 1
    return None


def multiple_choice_score(prediction, references, choices):
    if exact_score(prediction, references):
        return 1.0
    pred_index = option_index(prediction, choices)
    if pred_index is None:
        return 0.0
    return float(
        any(option_index(reference, choices) == pred_index for reference in references)
    )


def score_row(row):
    prediction = str(row.get("pred") or row.get("text") or "")
    references = references_for_row(row)
    metadata = row.get("metadata") or {}
    dataset = str(row.get("dataset") or metadata.get("dataset") or "").lower()
    choices = normalize_choices(metadata.get("choices"))
    exact = exact_score(prediction, references)
    token_f1 = token_f1_score(prediction, references)

    if choices:
        native = multiple_choice_score(prediction, references, choices)
        metric = "multiple_choice_accuracy"
    elif dataset in ANLS_DATASETS:
        native = anls_score(prediction, references)
        metric = "anls_single_reference"
    elif dataset in RELAXED_NUMERIC_DATASETS:
        native = numeric_relaxed_score(prediction, references)
        metric = "relaxed_numeric_accuracy"
    else:
        native = exact
        metric = "normalized_exact_single_reference"

    return {
        "native_proxy": native,
        "exact_match": exact,
        "token_f1": token_f1,
        "metric": metric,
        "dataset": dataset,
        "has_reference": bool(references),
    }


def mean(values):
    return sum(values) / len(values) if values else None


def discover_cells(eval_root, categories, allow_incomplete):
    cells = {}
    stage_dirs = []
    for stage_dir in sorted(eval_root.iterdir()):
        if not stage_dir.is_dir():
            continue
        match = STAGE_RE.match(stage_dir.name)
        if not match:
            continue
        stage_index = int(match.group(1))
        predictions_dir = stage_dir / "predictions"
        if predictions_dir.is_dir():
            stage_dirs.append((stage_index, stage_dir, predictions_dir))

    if not stage_dirs:
        raise FileNotFoundError("No stage prediction directories found under {}".format(eval_root))

    missing = []
    metric_usage = Counter()
    dataset_usage = Counter()
    total_rows = 0
    for stage_index, stage_dir, predictions_dir in stage_dirs:
        for category in categories:
            path = predictions_dir / (category + ".jsonl")
            if not path.is_file():
                missing.append("{}/{}".format(stage_dir.name, category))
                continue
            rows = read_jsonl(path)
            scored = [score_row(row) for row in rows]
            no_reference = sum(not item["has_reference"] for item in scored)
            if no_reference:
                raise ValueError(
                    "{} contains {} predictions without references".format(path, no_reference)
                )
            for item in scored:
                metric_usage[item["metric"]] += 1
                dataset_usage[item["dataset"]] += 1
            total_rows += len(rows)
            cells[(stage_index, category)] = {
                "stage_index": stage_index,
                "category": category,
                "prediction_file": str(path.resolve()),
                "samples": len(rows),
                "native_proxy": mean([item["native_proxy"] for item in scored]),
                "exact_match": mean([item["exact_match"] for item in scored]),
                "token_f1": mean([item["token_f1"] for item in scored]),
            }

    if missing and not allow_incomplete:
        raise RuntimeError(
            "Prediction matrix is incomplete ({} cells). First missing: {}".format(
                len(missing), ", ".join(missing[:10])
            )
        )
    return cells, sorted({index for index, _, _ in stage_dirs}), {
        "total_predictions": total_rows,
        "missing_cells": missing,
        "metric_usage": dict(sorted(metric_usage.items())),
        "dataset_usage": dict(sorted(dataset_usage.items())),
    }


def matrix_for_metric(cells, stage_indices, categories, metric):
    matrix = {}
    for stage_index in stage_indices:
        matrix[stage_index] = {}
        for category in categories:
            cell = cells.get((stage_index, category))
            matrix[stage_index][category] = cell.get(metric) if cell else None
    return matrix


def write_matrix(path, matrix, categories):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", *categories, "mean_all", "mean_seen"])
        for stage_index in sorted(matrix):
            values = [matrix[stage_index].get(category) for category in categories]
            all_values = [value for value in values if value is not None]
            seen_values = [
                value
                for index, value in enumerate(values, start=1)
                if index <= stage_index and value is not None
            ]
            writer.writerow(
                [
                    stage_index,
                    *["" if value is None else "{:.8f}".format(value) for value in values],
                    "" if not all_values else "{:.8f}".format(mean(all_values)),
                    "" if not seen_values else "{:.8f}".format(mean(seen_values)),
                ]
            )


def continual_metrics(matrix, categories):
    final_stage = max(matrix)
    final_scores = [
        matrix[final_stage].get(category)
        for category in categories
        if matrix[final_stage].get(category) is not None
    ]
    seen_averages = []
    for stage_index in sorted(matrix):
        values = [
            matrix[stage_index].get(category)
            for index, category in enumerate(categories, start=1)
            if index <= stage_index and matrix[stage_index].get(category) is not None
        ]
        if values:
            seen_averages.append(mean(values))

    per_task = []
    bwt_terms = []
    forgetting_terms = []
    for task_index, category in enumerate(categories, start=1):
        final_score = matrix[final_stage].get(category)
        acquisition = matrix.get(task_index, {}).get(category)
        trajectory = [
            matrix[stage].get(category)
            for stage in sorted(matrix)
            if stage >= task_index and matrix[stage].get(category) is not None
        ]
        forgetting = (
            max(trajectory) - final_score
            if trajectory and final_score is not None
            else None
        )
        backward = (
            final_score - acquisition
            if final_score is not None and acquisition is not None
            else None
        )
        per_task.append(
            {
                "task_index": task_index,
                "category": category,
                "acquisition_score": acquisition,
                "best_post_acquisition_score": max(trajectory) if trajectory else None,
                "final_score": final_score,
                "forgetting": forgetting,
                "backward_transfer": backward,
            }
        )
        if task_index < len(categories) and backward is not None:
            bwt_terms.append(backward)
        if task_index < len(categories) and forgetting is not None:
            forgetting_terms.append(forgetting)

    forward_proxy = []
    for task_index, category in enumerate(categories, start=1):
        if task_index == 1:
            continue
        value = matrix.get(task_index - 1, {}).get(category)
        if value is not None:
            forward_proxy.append(value)

    return {
        "score_scale": "[0,1]",
        "final_average_accuracy": mean(final_scores),
        "average_incremental_accuracy": mean(seen_averages),
        "backward_transfer": mean(bwt_terms),
        "average_forgetting": mean(forgetting_terms),
        "pre_learning_accuracy_proxy": mean(forward_proxy),
        "per_task": per_task,
    }


def write_forgetting_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "task_index",
            "category",
            "acquisition_score",
            "best_post_acquisition_score",
            "final_score",
            "forgetting",
            "backward_transfer",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_value(value):
    return "n/a" if value is None else "{:.4f}".format(value)


def write_markdown(path, payload):
    native = payload["continual_metrics"]["native_proxy"]
    exact = payload["continual_metrics"]["exact_match"]
    f1 = payload["continual_metrics"]["token_f1"]
    lines = [
        "# CoIN++ Continual-Learning Summary",
        "",
        "- Method: {}".format(payload["method"]),
        "- Profile: {}".format(payload["profile"]),
        "- Factor: {}".format(payload["factor"]),
        "- Predictions: {}".format(payload["audit"]["total_predictions"]),
        "- Score scale: [0,1]",
        "",
        "## Main Results",
        "",
        "| Metric | Native proxy | Exact match | Token F1 |",
        "|---|---:|---:|---:|",
        "| Final average | {} | {} | {} |".format(
            format_value(native["final_average_accuracy"]),
            format_value(exact["final_average_accuracy"]),
            format_value(f1["final_average_accuracy"]),
        ),
        "| Average incremental accuracy | {} | {} | {} |".format(
            format_value(native["average_incremental_accuracy"]),
            format_value(exact["average_incremental_accuracy"]),
            format_value(f1["average_incremental_accuracy"]),
        ),
        "| Backward transfer | {} | {} | {} |".format(
            format_value(native["backward_transfer"]),
            format_value(exact["backward_transfer"]),
            format_value(f1["backward_transfer"]),
        ),
        "| Average forgetting | {} | {} | {} |".format(
            format_value(native["average_forgetting"]),
            format_value(exact["average_forgetting"]),
            format_value(f1["average_forgetting"]),
        ),
        "",
        "The native proxy applies multiple-choice accuracy, single-reference ANLS, "
        "relaxed numeric accuracy, or normalized exact match according to the "
        "available dataset metadata. It is not labeled as an official metric when "
        "the original benchmark annotations are unavailable.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.result_root = args.result_root.resolve()
    eval_root = (
        args.eval_root.resolve()
        if args.eval_root
        else args.result_root / args.profile / args.factor / args.method / "eval"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else eval_root / "summary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(
        args.dataset_root / args.profile / args.factor / "manifest.json"
    )
    categories = list(manifest["trainable_order"])
    cells, stage_indices, audit = discover_cells(
        eval_root, categories, args.allow_incomplete
    )

    matrices = {
        metric: matrix_for_metric(cells, stage_indices, categories, metric)
        for metric in ("native_proxy", "exact_match", "token_f1")
    }
    continual = {
        metric: continual_metrics(matrix, categories)
        for metric, matrix in matrices.items()
    }
    payload = {
        "benchmark": "CoIN++",
        "method": args.method,
        "profile": args.profile,
        "factor": args.factor,
        "categories": categories,
        "eval_root": str(eval_root),
        "audit": audit,
        "continual_metrics": continual,
        "cells": [
            cells[key]
            for key in sorted(cells, key=lambda item: (item[0], categories.index(item[1])))
        ],
    }

    for metric, matrix in matrices.items():
        write_matrix(output_dir / ("matrix_" + metric + ".csv"), matrix, categories)
    write_forgetting_csv(
        output_dir / "forgetting_native_proxy.csv",
        continual["native_proxy"]["per_task"],
    )
    write_json(output_dir / "summary.json", payload)
    write_markdown(output_dir / "summary.md", payload)

    native = continual["native_proxy"]
    print("CoIN++ continual summary")
    print("  method/profile/factor: {} / {} / {}".format(args.method, args.profile, args.factor))
    print("  predictions:           {}".format(audit["total_predictions"]))
    print("  final average:         {}".format(format_value(native["final_average_accuracy"])))
    print("  BWT:                   {}".format(format_value(native["backward_transfer"])))
    print("  average forgetting:    {}".format(format_value(native["average_forgetting"])))
    print("  output:                {}".format(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
