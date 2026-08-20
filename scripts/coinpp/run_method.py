#!/usr/bin/env python3
"""Generate and run MCITlib LLaVA continual-training jobs on CoIN++."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    "LoRA-FT": {"workflow": "sequential", "rank": 32, "batch_size": 4},
    "Replay": {"workflow": "sequential", "rank": 32, "batch_size": 4},
    "OLoRA": {"workflow": "sequential", "rank_mode": "expert_scaled", "batch_size": 4},
    "MoELoRA": {"workflow": "sequential", "rank": 32, "batch_size": 4},
    "ModalPrompt": {"workflow": "sequential", "rank": 0, "batch_size": 8},
    "CL-MoE": {"workflow": "sequential", "rank_mode": "expert_scaled", "batch_size": 8},
    "HiDe": {"workflow": "sequential", "rank_mode": "expert_scaled", "batch_size": 8},
    "RegLoRA": {"workflow": "sequential_merged", "rank": 32, "batch_size": 4},
    "DISCO": {"workflow": "sequential", "rank_mode": "expert_scaled", "batch_size": 8},
    "SMoLoRA": {"workflow": "sequential", "rank_mode": "smolora", "batch_size": 4},
    "MR-LoRA": {"workflow": "independent_router", "rank": 32, "batch_size": 4},
    "KeepLoRA": {"workflow": "keep_lora", "rank": 64, "batch_size": 8},
}
FACTORS = ("visual_substrate", "skill_requirement", "evidence_complexity")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
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
        "--dataset-root", type=Path, default=REPO_ROOT / "datasets" / "CoIN++"
    )
    parser.add_argument(
        "--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints" / "CoIN++"
    )
    parser.add_argument(
        "--result-root", type=Path, default=REPO_ROOT / "results" / "CoIN++"
    )
    parser.add_argument("--runtime-root", type=Path, default=REPO_ROOT / "runs" / "CoIN++")
    parser.add_argument("--gpu-num", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-acc", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--expert-num", type=int)
    parser.add_argument("--prefix-len", type=int, default=20)
    parser.add_argument("--gradient-checkpointing", choices=("True", "False"), default="True")
    parser.add_argument("--smolora-emb", type=Path)
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--stop-stage", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--install-editable",
        action="store_true",
        help="Run pip install -e . once in the selected method directory.",
    )
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


def write_runtime_json(path, value, dry_run):
    """Keep command-only dry runs free of filesystem side effects."""
    if not dry_run:
        write_json(path, value)


def shell_join(command):
    return " ".join(shlex.quote(str(part)) for part in command)


def scaled_rank(num_stages):
    per_expert = 32 if num_stages <= 5 else 16
    return per_expert * num_stages


def method_rank(args, num_stages):
    if args.rank is not None:
        return args.rank
    settings = METHODS[args.method]
    mode = settings.get("rank_mode")
    if mode == "expert_scaled":
        return scaled_rank(num_stages)
    if mode == "smolora":
        return 16 * num_stages
    return settings["rank"]


def previous_checkpoint(method, output_dir):
    if method in {"RegLoRA", "KeepLoRA"}:
        return Path(str(output_dir) + "_merged")
    return output_dir


def checkpoint_name(method, stage_index, category):
    """Keep MCITlib's loader-friendly llava/lora checkpoint naming."""
    suffix = "llava" if method == "ModalPrompt" else "llava_lora"
    return "task{:02d}_{}_{}".format(stage_index, category, suffix)


def check_model_config(path):
    data = load_json(path)
    required = ("model_name", "mm_projector", "vision_tower")
    missing_keys = [key for key in required if not data.get(key)]
    missing_paths = [data[key] for key in required if data.get(key) and not Path(data[key]).exists()]
    if missing_keys or missing_paths:
        raise FileNotFoundError(
            "Invalid model config {}: missing keys={} missing paths={}".format(
                path, missing_keys, missing_paths
            )
        )
    return data


def stage_config(
    args,
    category,
    stage_index,
    num_stages,
    output_dir,
    previous_model,
    rank,
    expert_num,
):
    settings = METHODS[args.method]
    config = {
        "benchmark": "CoIN++",
        "profile": args.profile,
        "factor": args.factor,
        "category": category,
        "stage_index": stage_index,
        "gpu_num": args.gpu_num,
        "rank": rank,
        "expert_num": expert_num,
        "output_dir": str(output_dir),
        "epoch": args.epochs,
        "batch_size": args.batch_size or settings["batch_size"],
        "grad_acc": args.grad_acc,
        "lr": args.lr,
        "cur_task": stage_index if args.method == "ModalPrompt" else stage_index - 1,
        "num_tasks": num_stages,
        "prefix_len": args.prefix_len,
        "task": "CoINPP_{}_{}".format(args.factor, category),
        "gradient_checkpointing": args.gradient_checkpointing,
        "ins_type": stage_index - 1,
        "base_model": "llava",
        "key_id": max(0, stage_index - 2),
        "key_path": str(output_dir.parent),
    }
    if previous_model is not None:
        config["previous_model"] = str(previous_model)
    if args.smolora_emb:
        config["ins_emb"] = str(args.smolora_emb.resolve())
    return config


def run_logged(command, cwd, log_path, dry_run, env=None):
    print("$ (cd {} && {})".format(cwd, shell_join(command)))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n[{}] {}\n".format(datetime.now().isoformat(), shell_join(command)))
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(cwd),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def completion_path(output_dir):
    return output_dir / ".coinpp_complete.json"


def is_complete(output_dir):
    return completion_path(output_dir).is_file()


def mark_complete(output_dir, payload, dry_run):
    if dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(completion_path(output_dir), payload)


def standard_stage_command(method_dir, stage_index, model_config, data_config, train_config):
    script_name = "Task1.sh" if stage_index == 1 else "Taskn.sh"
    return [
        "bash",
        method_dir / "scripts" / "MCITlib" / "Train" / script_name,
        model_config,
        data_config,
        train_config,
    ]


def train_sequential(args, categories, model_config, method_dir, runtime_dir, output_root, log_root):
    num_stages = len(categories)
    rank = method_rank(args, num_stages)
    expert_num = args.expert_num or num_stages
    previous = None

    if args.method == "SMoLoRA":
        if not args.smolora_emb:
            if not args.dry_run:
                raise ValueError("SMoLoRA requires --smolora-emb generated for this factor")
            print("[dry-run] SMoLoRA embedding path was not supplied; using a placeholder")
            args.smolora_emb = runtime_dir / "smolora_embeddings.placeholder.pkl"
        if not args.dry_run and not args.smolora_emb.is_file():
            raise FileNotFoundError("SMoLoRA embedding file not found: {}".format(args.smolora_emb))

    for stage_index, category in enumerate(categories, start=1):
        if stage_index < args.start_stage:
            previous = previous_checkpoint(
                args.method,
                output_root / checkpoint_name(args.method, stage_index, category),
            )
            continue
        if args.stop_stage and stage_index > args.stop_stage:
            break

        output_dir = output_root / checkpoint_name(args.method, stage_index, category)
        train_config_path = runtime_dir / "train" / "task{:02d}_{}.json".format(
            stage_index, category
        )
        config = stage_config(
            args,
            category,
            stage_index,
            num_stages,
            output_dir,
            previous,
            rank,
            expert_num,
        )
        write_runtime_json(train_config_path, config, args.dry_run)
        data_config = (
            args.data_config_root
            / args.profile
            / args.factor
            / (category + ".json")
        )
        if not data_config.is_file():
            raise FileNotFoundError("Data config not found: {}".format(data_config))

        if is_complete(output_dir) and not args.force:
            print("[skip] {} stage {} {} is complete".format(args.method, stage_index, category))
        else:
            command = standard_stage_command(
                method_dir,
                stage_index,
                model_config,
                data_config,
                train_config_path,
            )
            run_logged(
                command,
                method_dir,
                log_root / "task{:02d}_{}.log".format(stage_index, category),
                args.dry_run,
            )
            mark_complete(
                output_dir,
                {
                    "method": args.method,
                    "factor": args.factor,
                    "category": category,
                    "stage_index": stage_index,
                    "train_config": str(train_config_path),
                    "data_config": str(data_config),
                },
                args.dry_run,
            )
        previous = previous_checkpoint(args.method, output_dir)


def train_keep_lora(args, categories, model_config, method_dir, runtime_dir, output_root, log_root):
    num_stages = len(categories)
    rank = method_rank(args, num_stages)
    initial_space = output_root / "task00_base_gradients"
    initial_config = runtime_dir / "train_pre" / "task00.json"
    write_runtime_json(
        initial_config, {"output_dir": str(initial_space)}, args.dry_run
    )
    initial_command = [
        "bash",
        method_dir / "scripts" / "MCITlib" / "Train" / "extract_weights.sh",
        model_config,
        initial_config,
        "0.6",
    ]
    if not (initial_space / "gradients_merged.pt").is_file() or args.force:
        run_logged(
            initial_command,
            method_dir,
            log_root / "task00_extract_weights.log",
            args.dry_run,
        )

    previous_model = None
    previous_space = initial_space
    for stage_index, category in enumerate(categories, start=1):
        output_dir = output_root / checkpoint_name("KeepLoRA", stage_index, category)
        merged_model = previous_checkpoint("KeepLoRA", output_dir)
        gradient_dir = Path(str(output_dir) + "_gradients")
        if stage_index < args.start_stage:
            previous_model = merged_model
            previous_space = gradient_dir
            continue
        if args.stop_stage and stage_index > args.stop_stage:
            break

        data_config = (
            args.data_config_root
            / args.profile
            / args.factor
            / (category + ".json")
        )
        pre_config = runtime_dir / "train_pre" / "task{:02d}.json".format(stage_index)
        train_config = runtime_dir / "train" / "task{:02d}.json".format(stage_index)
        post_config = runtime_dir / "train_post" / "task{:02d}.json".format(stage_index)
        model_path = (
            load_json(model_config)["model_name"] if stage_index == 1 else str(previous_model)
        )
        write_runtime_json(
            pre_config,
            {
                "gpu_num": args.gpu_num,
                "stage": "KeepLoRA-task{}".format(stage_index),
                "model_path": model_path,
                "rank": rank,
                "space_path": str(previous_space),
                "output_dir": str(gradient_dir),
            },
            args.dry_run,
        )
        train_payload = stage_config(
            args,
            category,
            stage_index,
            num_stages,
            output_dir,
            previous_model,
            rank,
            num_stages,
        )
        write_runtime_json(train_config, train_payload, args.dry_run)
        write_runtime_json(
            post_config,
            {
                "gpu_num": args.gpu_num,
                "stage": "KeepLoRA-task{}".format(stage_index),
                "model_path": str(merged_model),
                "space_path": str(previous_space),
                "energy_threshold": 0.99,
                "output_dir": str(gradient_dir),
            },
            args.dry_run,
        )

        if is_complete(output_dir) and not args.force:
            print("[skip] KeepLoRA stage {} {} is complete".format(stage_index, category))
        else:
            pre_command = [
                "bash",
                method_dir / "scripts" / "MCITlib" / "Train" / "extract_gradients.sh",
                model_config,
                data_config,
                pre_config,
                "0.2",
                "fixed_rank",
            ]
            task_command = standard_stage_command(
                method_dir,
                stage_index,
                model_config,
                data_config,
                train_config,
            )
            post_command = [
                "bash",
                method_dir / "scripts" / "MCITlib" / "Train" / "extract_gradients.sh",
                model_config,
                data_config,
                post_config,
                "0.2",
                "energy",
            ]
            log_path = log_root / "task{:02d}_{}.log".format(stage_index, category)
            run_logged(pre_command, method_dir, log_path, args.dry_run)
            run_logged(task_command, method_dir, log_path, args.dry_run)
            run_logged(post_command, method_dir, log_path, args.dry_run)
            mark_complete(
                output_dir,
                {
                    "method": "KeepLoRA",
                    "factor": args.factor,
                    "category": category,
                    "stage_index": stage_index,
                },
                args.dry_run,
            )
        previous_model = merged_model
        previous_space = gradient_dir


def train_mr_lora(args, categories, model_config, method_dir, runtime_dir, output_root, log_root):
    num_stages = len(categories)
    rank = method_rank(args, num_stages)
    for stage_index, category in enumerate(categories, start=1):
        if stage_index < args.start_stage:
            continue
        if args.stop_stage and stage_index > args.stop_stage:
            break
        data_config = (
            args.data_config_root
            / args.profile
            / args.factor
            / (category + ".json")
        )
        expert_output = output_root / "experts" / checkpoint_name(
            "MR-LoRA", stage_index, category
        )
        router_output = output_root / "routers" / checkpoint_name(
            "MR-LoRA", stage_index, category
        )
        expert_config = runtime_dir / "experts" / "task{:02d}.json".format(stage_index)
        router_config = runtime_dir / "routers" / "task{:02d}.json".format(stage_index)
        base = {
            "gpu_num": args.gpu_num,
            "rank": rank,
            "epoch": args.epochs,
            "batch_size": args.batch_size or METHODS["MR-LoRA"]["batch_size"],
            "grad_acc": args.grad_acc,
            "lr": args.lr,
        }
        write_runtime_json(
            expert_config,
            dict(base, output_dir=str(expert_output)),
            args.dry_run,
        )
        write_runtime_json(
            router_config,
            dict(base, output_dir=str(router_output)),
            args.dry_run,
        )
        if is_complete(expert_output) and is_complete(router_output) and not args.force:
            print("[skip] MR-LoRA expert/router {} is complete".format(category))
            continue
        expert_command = [
            "bash",
            method_dir / "scripts" / "MCITlib" / "Train" / "Task1.sh",
            model_config,
            data_config,
            expert_config,
        ]
        router_command = [
            "bash",
            method_dir / "scripts" / "MCITlib" / "Train" / "Task1_router.sh",
            model_config,
            data_config,
            router_config,
        ]
        log_path = log_root / "task{:02d}_{}.log".format(stage_index, category)
        run_logged(expert_command, method_dir, log_path, args.dry_run)
        run_logged(router_command, method_dir, log_path, args.dry_run)
        mark_complete(
            expert_output,
            {
                "method": "MR-LoRA",
                "component": "expert",
                "factor": args.factor,
                "category": category,
            },
            args.dry_run,
        )
        mark_complete(
            router_output,
            {
                "method": "MR-LoRA",
                "component": "router",
                "factor": args.factor,
                "category": category,
            },
            args.dry_run,
        )


def main():
    args = parse_args()
    args.model_config = args.model_config.resolve()
    args.data_config_root = args.data_config_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.result_root = args.result_root.resolve()
    args.runtime_root = args.runtime_root.resolve()
    check_model_config(args.model_config)

    manifest_path = args.dataset_root / args.profile / args.factor / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Prepared CoIN++ manifest not found: {}. Run prepare_coinpp.py first.".format(
                manifest_path
            )
        )
    categories = list(load_json(manifest_path)["trainable_order"])
    method_dir = REPO_ROOT / "LLaVA" / args.method
    if not method_dir.is_dir():
        raise FileNotFoundError("Method directory not found: {}".format(method_dir))

    output_root = (
        args.checkpoint_root / args.profile / args.factor / args.method
    ).resolve()
    log_root = (args.result_root / args.profile / args.factor / args.method / "logs").resolve()
    runtime_dir = (
        args.runtime_root / args.profile / args.factor / args.method
    ).resolve()
    if not args.dry_run:
        runtime_dir.mkdir(parents=True, exist_ok=True)

    if args.install_editable:
        pip = shutil.which("pip") or "pip"
        run_logged(
            [pip, "install", "-e", "."],
            method_dir,
            log_root / "install.log",
            args.dry_run,
        )

    settings = METHODS[args.method]
    run_manifest = {
        "benchmark": "CoIN++",
        "method": args.method,
        "factor": args.factor,
        "profile": args.profile,
        "workflow": settings["workflow"],
        "categories": categories,
        "model_config": str(args.model_config),
        "checkpoint_root": str(output_root),
        "checkpoint_pattern": checkpoint_name(args.method, 1, "CATEGORY").replace(
            "task01", "task{stage:02d}"
        ),
        "result_root": str(log_root.parent),
        "dry_run": args.dry_run,
    }
    write_runtime_json(
        runtime_dir / "run_manifest.json", run_manifest, args.dry_run
    )

    print("MCITlib CoIN++ training")
    print("  method/profile/factor: {} / {} / {}".format(args.method, args.profile, args.factor))
    print("  workflow:              {}".format(settings["workflow"]))
    print("  stages:                {}".format(len(categories)))
    print("  checkpoints:           {}".format(output_root))
    print("  runtime configs:       {}".format(runtime_dir))

    if settings["workflow"] in {"sequential", "sequential_merged"}:
        train_sequential(
            args,
            categories,
            args.model_config,
            method_dir,
            runtime_dir,
            output_root,
            log_root,
        )
    elif settings["workflow"] == "keep_lora":
        train_keep_lora(
            args,
            categories,
            args.model_config,
            method_dir,
            runtime_dir,
            output_root,
            log_root,
        )
    elif settings["workflow"] == "independent_router":
        train_mr_lora(
            args,
            categories,
            args.model_config,
            method_dir,
            runtime_dir,
            output_root,
            log_root,
        )
    else:
        raise NotImplementedError(settings["workflow"])

    print("CoIN++ method run completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
