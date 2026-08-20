#!/usr/bin/env python3
"""Evaluate every checkpoint/task cell of a CoIN++ continual-learning run."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS = (
    "LoRA-FT",
    "Replay",
    "OLoRA",
    "MoELoRA",
    "ModalPrompt",
    "CL-MoE",
    "HiDe",
    "RegLoRA",
    "DISCO",
    "SMoLoRA",
    "KeepLoRA",
    "MR-LoRA",
)
FACTORS = ("visual_substrate", "skill_requirement", "evidence_complexity")
MERGED_METHODS = {"RegLoRA", "KeepLoRA"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--factor", choices=FACTORS, required=True)
    parser.add_argument("--profile", choices=("clean_5k", "strict", "legacy_10k"), default="strict")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "configs" / "model_configs" / "llava_coinpp_local.json",
    )
    parser.add_argument(
        "--data-config-root",
        type=Path,
        default=REPO_ROOT / "configs" / "data_configs" / "CoIN++",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "CoIN++",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "CoIN++",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=REPO_ROOT / "results" / "CoIN++",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=REPO_ROOT / "runs" / "CoIN++",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--stop-stage", type=int, default=0)
    parser.add_argument("--eval-scope", choices=("all", "seen"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prefix-len", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-missing-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def checkpoint_name(method, stage_index, category):
    suffix = "llava" if method == "ModalPrompt" else "llava_lora"
    value = "task{:02d}_{}_{}".format(stage_index, category, suffix)
    if method in MERGED_METHODS:
        value += "_merged"
    return value


def count_nonempty_lines(path):
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def expected_rows(question_file, limit):
    rows = load_json(question_file)
    return min(len(rows), limit) if limit is not None else len(rows)


def shell_join(command):
    return " ".join(shlex.quote(str(part)) for part in command)


def run_logged(command, cwd, log_path, gpu, dry_run):
    print("$ CUDA_VISIBLE_DEVICES={} {}".format(gpu, shell_join(command)))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def validate_model_config(path):
    config = load_json(path)
    required = ("model_name", "mm_projector", "vision_tower")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError("Model config is missing keys: {}".format(missing))
    for key in required:
        if not Path(config[key]).exists():
            raise FileNotFoundError("{} does not exist: {}".format(key, config[key]))
    return config


def build_stage_jobs(args, categories, stage_index, stage_dir):
    eval_categories = (
        categories if args.eval_scope == "all" else categories[:stage_index]
    )
    jobs = []
    image_folders = set()
    for category in eval_categories:
        config_path = (
            args.data_config_root
            / args.profile
            / args.factor
            / (category + ".json")
        )
        data_config = load_json(config_path)
        question_file = Path(data_config["test_path"])
        image_folder = Path(data_config["test_folder"])
        output_file = stage_dir / "predictions" / (category + ".jsonl")
        image_folders.add(str(image_folder.resolve()))
        expected = expected_rows(question_file, args.limit)
        existing = count_nonempty_lines(output_file)
        if existing == expected and not args.force:
            print(
                "[skip] stage={} eval={} predictions={}".format(
                    stage_index, category, existing
                )
            )
            continue
        jobs.append(
            {
                "category": category,
                "question_file": str(question_file.resolve()),
                "answers_file": str(output_file.resolve()),
                "expected_rows": expected,
            }
        )
    if len(image_folders) > 1:
        raise ValueError(
            "A single inference stage requires one image root; found {}".format(
                sorted(image_folders)
            )
        )
    image_folder = Path(next(iter(image_folders))) if image_folders else None
    return jobs, image_folder


def main():
    args = parse_args()
    for field in (
        "model_config",
        "data_config_root",
        "dataset_root",
        "checkpoint_root",
        "result_root",
        "runtime_root",
    ):
        setattr(args, field, getattr(args, field).resolve())
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.method == "MR-LoRA":
        raise SystemExit(
            "MR-LoRA trains independent experts and a task router rather than a "
            "single sequential checkpoint. Use run_method.py to train it; routed "
            "CoIN++ matrix evaluation requires a factor-specific router mapping."
        )

    manifest_path = args.dataset_root / args.profile / args.factor / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Prepared dataset manifest not found: {}. Run prepare_coinpp.py.".format(
                manifest_path
            )
        )
    categories = list(load_json(manifest_path)["trainable_order"])
    num_tasks = len(categories)
    if args.start_stage < 1 or args.start_stage > num_tasks:
        raise ValueError("--start-stage is outside the task sequence")
    stop_stage = args.stop_stage or num_tasks
    stop_stage = min(stop_stage, num_tasks)

    model_config = validate_model_config(args.model_config)
    method_root = REPO_ROOT / "LLaVA" / args.method
    checkpoint_base = (
        args.checkpoint_root / args.profile / args.factor / args.method
    )
    eval_root = args.result_root / args.profile / args.factor / args.method / "eval"
    runtime_root = (
        args.runtime_root / args.profile / args.factor / args.method / "eval"
    )
    if not args.dry_run:
        runtime_root.mkdir(parents=True, exist_ok=True)

    print("MCITlib CoIN++ evaluation")
    print("  method/profile/factor: {} / {} / {}".format(args.method, args.profile, args.factor))
    print("  stages:                {}-{}".format(args.start_stage, stop_stage))
    print("  eval scope:            {}".format(args.eval_scope))
    print("  result root:           {}".format(eval_root))

    for stage_index in range(args.start_stage, stop_stage + 1):
        train_category = categories[stage_index - 1]
        checkpoint = checkpoint_base / checkpoint_name(
            args.method, stage_index, train_category
        )
        if not args.dry_run and not checkpoint.is_dir():
            message = "Checkpoint not found: {}".format(checkpoint)
            if args.skip_missing_checkpoints:
                print("[skip] " + message)
                continue
            raise FileNotFoundError(message)

        stage_name = "{}_{}".format(stage_index, train_category)
        stage_dir = eval_root / stage_name
        jobs, image_folder = build_stage_jobs(
            args, categories, stage_index, stage_dir
        )
        if not jobs:
            print("[skip] {} is fully evaluated".format(stage_name))
            continue

        jobs_path = runtime_root / stage_name / "jobs.json"
        if not args.dry_run:
            write_json(jobs_path, jobs)
        command = [
            args.python,
            REPO_ROOT / "scripts" / "coinpp" / "model_vqa_coinpp.py",
            "--method",
            args.method,
            "--method-root",
            method_root,
            "--model-path",
            checkpoint,
            "--jobs-file",
            jobs_path,
            "--image-folder",
            image_folder,
            "--stage-index",
            stage_index,
            "--num-tasks",
            num_tasks,
            "--prefix-len",
            args.prefix_len,
            "--text-tower",
            model_config["vision_tower"],
            "--mm-projector",
            model_config["mm_projector"],
            "--max-new-tokens",
            args.max_new_tokens,
        ]
        if args.method not in MERGED_METHODS:
            command.extend(["--model-base", model_config["model_name"]])
        if args.limit is not None:
            command.extend(["--limit", args.limit])

        run_logged(
            command,
            method_root,
            stage_dir / "inference.log",
            args.gpu,
            args.dry_run,
        )
        if not args.dry_run:
            incomplete = []
            for job in jobs:
                output = Path(job["answers_file"])
                found = count_nonempty_lines(output)
                if found != job["expected_rows"]:
                    incomplete.append(
                        "{}={}/{}".format(
                            job["category"], found, job["expected_rows"]
                        )
                    )
            if incomplete:
                raise RuntimeError(
                    "Incomplete predictions for {}: {}".format(
                        stage_name, ", ".join(incomplete)
                    )
                )
            write_json(
                stage_dir / "stage_manifest.json",
                {
                    "benchmark": "CoIN++",
                    "method": args.method,
                    "profile": args.profile,
                    "factor": args.factor,
                    "stage_index": stage_index,
                    "train_category": train_category,
                    "checkpoint": str(checkpoint),
                    "predictions": jobs,
                },
            )

    print("CoIN++ evaluation completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
