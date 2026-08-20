#!/usr/bin/env python3
"""Build factor-level instruction embeddings required by MCITlib SMoLoRA."""

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FACTORS = ("visual_substrate", "skill_requirement", "evidence_complexity")
IMAGE_TOKEN_RE = re.compile(r"<image(?:\s+\d+)?>|<image>", flags=re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", choices=FACTORS, required=True)
    parser.add_argument("--profile", choices=("clean_5k", "strict", "legacy_10k"), default="strict")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "CoIN++",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples-per-category", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def instruction_for_row(row):
    for turn in row.get("conversations") or []:
        if str(turn.get("from", "")).lower() in {"human", "user"}:
            value = IMAGE_TOKEN_RE.sub("", str(turn.get("value") or ""))
            return " ".join(value.split())
    return ""


def stable_prompts(rows, limit):
    candidates = []
    for row in rows:
        prompt = instruction_for_row(row)
        if not prompt:
            continue
        sample_id = str(row.get("id") or prompt)
        key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
        candidates.append((key, prompt))
    candidates.sort(key=lambda item: item[0])
    return [prompt for _, prompt in candidates[:limit]]


def mean_pool(model_output, attention_mask, torch):
    embeddings = model_output.last_hidden_state
    expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    return (embeddings * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(min=1e-9)


def encode_prompts(prompts, tokenizer, model, batch_size, max_length, device, torch):
    output = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            model_output = model(**encoded)
        output.append(
            mean_pool(model_output, encoded["attention_mask"], torch).detach().cpu()
        )
    return torch.cat(output, dim=0)


def main():
    args = parse_args()
    if args.samples_per_category <= 0 or args.batch_size <= 0:
        raise ValueError("sample and batch sizes must be positive")

    import torch
    from transformers import AutoModel, AutoTokenizer

    dataset_root = args.dataset_root.resolve()
    factor_root = dataset_root / args.profile / args.factor
    manifest = load_json(factor_root / "manifest.json")
    categories = list(manifest["trainable_order"])
    output = (
        args.output.resolve()
        if args.output
        else REPO_ROOT
        / "datasets"
        / "CoIN++"
        / "embeddings"
        / args.profile
        / (args.factor + "_smolora.pkl")
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path.resolve()))
    model = AutoModel.from_pretrained(str(args.model_path.resolve())).to(device)
    model.eval()

    category_embeddings = []
    report = []
    for category in categories:
        rows = load_json(factor_root / category / "train.json")
        prompts = stable_prompts(rows, args.samples_per_category)
        if not prompts:
            raise ValueError("No usable prompts for category {}".format(category))
        embeddings = encode_prompts(
            prompts,
            tokenizer,
            model,
            args.batch_size,
            args.max_length,
            device,
            torch,
        )
        category_embeddings.append(embeddings.mean(dim=0, keepdim=True))
        report.append({"category": category, "prompts": len(prompts)})
        print("{}: prompts={}".format(category, len(prompts)))

    tensor = torch.cat(category_embeddings, dim=0)
    with output.open("wb") as handle:
        pickle.dump(tensor, handle)
    manifest_path = output.with_suffix(output.suffix + ".json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "benchmark": "CoIN++",
                "profile": args.profile,
                "factor": args.factor,
                "model_path": str(args.model_path.resolve()),
                "shape": list(tensor.shape),
                "categories": report,
                "output": str(output),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    print("Wrote SMoLoRA embeddings: {} shape={}".format(output, tuple(tensor.shape)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
