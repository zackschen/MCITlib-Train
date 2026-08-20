#!/usr/bin/env python3
"""Prepare CoIN++ factor streams for MCITlib's LLaVA methods."""

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(
    "/home/chencheng/data/Code/Easy_Train_MLLM/cl_dataset/coin_factor1_final"
)
DEFAULT_MEDIA = Path("/home/chencheng/data/Code/Easy_Train_MLLM/cl_dataset")
DEFAULT_CLEAN_SOURCE = Path(
    "/home/chencheng/data/Code/Easy_Train_MLLM/cl_dataset/coin_factor1_final_clean"
)
FACTORS = ("visual_substrate", "skill_requirement", "evidence_complexity")
IMAGE_TOKEN_RE = re.compile(r"<image(?:\s+\d+)?>", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA)
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "datasets" / "CoIN++"
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=REPO_ROOT / "configs" / "data_configs" / "CoIN++",
    )
    parser.add_argument(
        "--profile",
        choices=("clean_5k", "strict", "legacy_10k"),
        default="strict",
        help=(
            "clean_5k uses the image-backed IconQA-free final splits; strict samples "
            "image-backed rows from the legacy source; legacy_10k preserves source splits."
        ),
    )
    parser.add_argument("--factors", nargs="+", choices=FACTORS, default=list(FACTORS))
    parser.add_argument("--train-per-category", type=int, default=4000)
    parser.add_argument("--eval-per-category", type=int, default=500)
    parser.add_argument("--replay-per-previous-task", type=int, default=100)
    parser.add_argument("--router-per-task", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--force-links", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp.replace(path)


def replace_symlink(path, target, force=False):
    target = target.resolve()
    if path.is_symlink():
        current = path.resolve()
        if current == target:
            return
        if not force:
            raise RuntimeError(
                "Symlink {} points to {}, expected {}; pass --force-links".format(
                    path, current, target
                )
            )
        path.unlink()
    elif path.exists():
        if not force:
            raise RuntimeError(
                "{} exists and is not the expected symlink; pass --force-links".format(
                    path
                )
            )
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=True)


def sample_id(row):
    return str(row.get("id") or row.get("question_id") or "")


def question_text(row):
    for turn in row.get("conversations", []):
        if str(turn.get("from", "")).lower() in {"human", "user"}:
            return str(turn.get("value", "")).strip()
    return str(row.get("question") or row.get("text") or "").strip()


def answer_text(row):
    answers = []
    for turn in row.get("conversations", []):
        if str(turn.get("from", "")).lower() in {"gpt", "assistant"}:
            value = str(turn.get("value", "")).strip()
            if value:
                answers.append(value)
    if answers:
        return answers[-1]
    value = row.get("answer")
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "").strip()


def clean_question(text):
    text = IMAGE_TOKEN_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def resolved_image(row, media_root):
    image = row.get("image")
    if not image:
        return None
    path = Path(str(image))
    if not path.is_absolute():
        path = media_root / path
    return path if path.is_file() else None


def deterministic_take(rows, limit, seed_key):
    if limit is None or limit <= 0 or len(rows) <= limit:
        return list(rows)

    def rank(row):
        value = "{}|{}|{}".format(seed_key, sample_id(row), row.get("image", ""))
        return hashlib.sha256(value.encode("utf-8")).digest()

    return sorted(rows, key=rank)[:limit]


def validate_unique(rows, label):
    ids = [sample_id(row) for row in rows]
    missing = sum(not value for value in ids)
    duplicates = len(ids) - len(set(ids))
    if missing or duplicates:
        raise ValueError(
            "{} has missing_ids={} duplicate_ids={}".format(label, missing, duplicates)
        )


def source_paths(source_root, factor, category):
    base = source_root / "splits" / factor / "trainable"
    return (
        base / "train" / category / "train.json",
        base / "eval" / category / "eval.json",
    )


def build_questions(rows, media_root, factor, category, profile):
    questions = []
    for row in rows:
        image_path = resolved_image(row, media_root)
        metadata = row.get("metadata") or {}
        questions.append(
            {
                "question_id": sample_id(row),
                "image": row.get("image") if image_path else None,
                "text": clean_question(question_text(row)),
                "answer": answer_text(row),
                "answers": [answer_text(row)],
                "has_image": image_path is not None,
                "factor": factor,
                "category": category,
                "profile": profile,
                "dataset": metadata.get("dataset"),
                "answer_type": metadata.get("answer_type"),
                "metadata": metadata,
            }
        )
    return questions


def selected_rows(rows, profile, media_root, limit, seed_key):
    valid = [row for row in rows if resolved_image(row, media_root)]
    if profile == "clean_5k":
        if len(valid) != len(rows):
            raise ValueError(
                "{} clean_5k requires every row to have an image; valid={}/{}".format(
                    seed_key, len(valid), len(rows)
                )
            )
        return list(rows), 0
    if profile == "strict":
        if len(valid) < limit:
            raise ValueError(
                "{} requires {} image-backed rows but only {} are available".format(
                    seed_key, limit, len(valid)
                )
            )
        return deterministic_take(valid, limit, seed_key), len(rows) - len(valid)
    return deterministic_take(rows, limit=None, seed_key=seed_key), len(rows) - len(valid)


def build_router_rows(selected_train, categories, stage_index, factor, args):
    """Create cumulative factor-router supervision for MR-LoRA."""
    seen = categories[:stage_index]
    if len(seen) > 26:
        raise ValueError("MR-LoRA router supports at most 26 factor categories")

    expert_lines = [
        "{}. {}".format(chr(ord("A") + index), category.replace("_", " "))
        for index, category in enumerate(seen)
    ]
    instructions = (
        "You are a router for a continual multimodal model. Select the expert "
        "whose {} best matches the user's question and required visual evidence.\n"
        "Available experts:\n{}\n"
        "User question: {}\n"
        "Return only the expert letter."
    )
    rows = []
    for index, category in enumerate(seen):
        code = chr(ord("A") + index)
        sampled = deterministic_take(
            selected_train[category],
            args.router_per_task,
            "{}|{}|router|{}".format(factor, category, args.seed),
        )
        for row in sampled:
            output = copy.deepcopy(row)
            question = clean_question(question_text(row))
            prompt = instructions.format(
                factor.replace("_", " "),
                "\n".join(expert_lines),
                question,
            )
            if output.get("image"):
                prompt = "<image>\n" + prompt
            output["conversations"] = [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": code},
            ]
            metadata = dict(output.get("metadata") or {})
            metadata.update(
                {
                    "router_factor": factor,
                    "router_category": category,
                    "router_label": code,
                    "router_seen_categories": list(seen),
                }
            )
            output["metadata"] = metadata
            rows.append(output)
    validate_unique(rows, "{}:router_stage{}".format(factor, stage_index))
    return rows



def build_factor(args, factor, profile_root, config_profile_root):
    order_file = args.source_root / "splits" / factor / "transition_order.json"
    order_data = load_json(order_file)
    categories = list(order_data["trainable_order"])
    selected_train = {}
    selected_eval = {}
    category_reports = []

    for category in categories:
        train_path, eval_path = source_paths(args.source_root, factor, category)
        train_rows = load_json(train_path)
        eval_rows = load_json(eval_path)
        validate_unique(train_rows, "{}:{}/source_train".format(factor, category))
        validate_unique(eval_rows, "{}:{}/source_eval".format(factor, category))

        train_limit = args.train_per_category if args.profile == "strict" else None
        eval_limit = args.eval_per_category if args.profile == "strict" else None
        train_rows_selected, train_missing = selected_rows(
            train_rows,
            args.profile,
            args.media_root,
            train_limit,
            "{}|{}|train|{}".format(factor, category, args.seed),
        )
        eval_rows_selected, eval_missing = selected_rows(
            eval_rows,
            args.profile,
            args.media_root,
            eval_limit,
            "{}|{}|eval|{}".format(factor, category, args.seed),
        )
        validate_unique(
            train_rows_selected, "{}:{}/selected_train".format(factor, category)
        )
        validate_unique(eval_rows_selected, "{}:{}/selected_eval".format(factor, category))
        overlap = set(map(sample_id, train_rows_selected)) & set(
            map(sample_id, eval_rows_selected)
        )
        if overlap:
            raise ValueError(
                "{}:{} has {} train/eval overlapping IDs".format(
                    factor, category, len(overlap)
                )
            )
        selected_train[category] = train_rows_selected
        selected_eval[category] = eval_rows_selected
        category_reports.append(
            {
                "category": category,
                "source_train": len(train_rows),
                "source_eval": len(eval_rows),
                "selected_train": len(train_rows_selected),
                "selected_eval": len(eval_rows_selected),
                "source_train_missing_image": train_missing,
                "source_eval_missing_image": eval_missing,
            }
        )

    for stage_index, category in enumerate(categories, start=1):
        stage_root = profile_root / factor / category
        train_out = stage_root / "train.json"
        eval_out = stage_root / "eval.json"
        questions_out = stage_root / "questions.json"
        replay_out = stage_root / "replay.json"
        router_out = stage_root / "router.json"

        if args.profile == "strict":
            write_json(train_out, selected_train[category])
            write_json(eval_out, selected_eval[category])
            train_config_path = train_out
            eval_annotation_path = eval_out
        else:
            train_config_path, eval_annotation_path = source_paths(
                args.source_root, factor, category
            )

        write_json(
            questions_out,
            build_questions(
                selected_eval[category],
                args.media_root,
                factor,
                category,
                args.profile,
            ),
        )

        replay_rows = list(selected_train[category])
        for previous in categories[: stage_index - 1]:
            replay_rows.extend(
                deterministic_take(
                    selected_train[previous],
                    args.replay_per_previous_task,
                    "{}|{}|{}|replay|{}".format(
                        factor, category, previous, args.seed
                    ),
                )
            )
        write_json(replay_out, replay_rows)
        write_json(
            router_out,
            build_router_rows(
                selected_train,
                categories,
                stage_index,
                factor,
                args,
            ),
        )

        data_config = {
            "benchmark": "CoIN++",
            "profile": args.profile,
            "factor": factor,
            "category": category,
            "stage_index": stage_index,
            "num_stages": len(categories),
            "train_path": str(train_config_path.resolve()),
            "replay_path": str(replay_out.resolve()),
            "router_path": str(router_out.resolve()),
            "test_path": str(questions_out.resolve()),
            "eval_annotation_path": str(eval_annotation_path.resolve()),
            "train_folder": str((args.output_root / "media").resolve()),
            "router_folder": str((args.output_root / "media").resolve()),
            "test_folder": str((args.output_root / "media").resolve()),
        }
        write_json(config_profile_root / factor / (category + ".json"), data_config)

    report = {
        "factor": factor,
        "field": order_data.get("field"),
        "profile": args.profile,
        "trainable_order": categories,
        "categories": category_reports,
    }
    write_json(profile_root / factor / "manifest.json", report)
    return report


def main():
    args = parse_args()
    if args.source_root is None:
        args.source_root = (
            DEFAULT_CLEAN_SOURCE if args.profile == "clean_5k" else DEFAULT_SOURCE
        )
    args.source_root = args.source_root.resolve()
    args.media_root = args.media_root.resolve()
    if not (args.source_root / "splits").is_dir():
        raise FileNotFoundError("CoIN++ source split root not found: {}".format(args.source_root))
    if not args.media_root.is_dir():
        raise FileNotFoundError("Media root not found: {}".format(args.media_root))

    profile_root = args.output_root / args.profile
    config_profile_root = args.config_root / args.profile
    if not args.validate_only:
        source_link = (
            args.output_root / "source_clean"
            if args.profile == "clean_5k"
            else args.output_root / "source"
        )
        replace_symlink(source_link, args.source_root, args.force_links)
        replace_symlink(args.output_root / "media", args.media_root, args.force_links)

    reports = []
    for factor in args.factors:
        reports.append(build_factor(args, factor, profile_root, config_profile_root))

    summary = {
        "benchmark": "CoIN++",
        "profile": args.profile,
        "source_root": str(args.source_root),
        "media_root": str(args.media_root),
        "seed": args.seed,
        "strict_train_per_category": args.train_per_category,
        "strict_eval_per_category": args.eval_per_category,
        "replay_per_previous_task": args.replay_per_previous_task,
        "router_per_task": args.router_per_task,
        "factors": reports,
    }
    write_json(profile_root / "manifest.json", summary)

    print("CoIN++ MCITlib data preparation completed")
    print("  profile: {}".format(args.profile))
    print("  data:    {}".format(profile_root))
    print("  configs: {}".format(config_profile_root))
    for report in reports:
        selected_train_total = sum(
            row["selected_train"] for row in report["categories"]
        )
        selected_eval_total = sum(row["selected_eval"] for row in report["categories"])
        missing_train = sum(
            row["source_train_missing_image"] for row in report["categories"]
        )
        print(
            "  {:20s} stages={:2d} train={:6d} eval={:5d} source_missing_images={:6d}".format(
                report["factor"],
                len(report["trainable_order"]),
                selected_train_total,
                selected_eval_total,
                missing_train,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
