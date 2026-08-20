#!/usr/bin/env python3
"""Run method-aware LLaVA inference for one CoIN++ continual-learning stage."""

import argparse
import importlib
import json
import math
import os
import re
import sys
import uuid
from pathlib import Path


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
)
IMAGE_TOKEN_RE = re.compile(r"<image(?:\s+\d+)?>|<image>", flags=re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--method-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-base", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--jobs-file", type=Path)
    parser.add_argument("--question-file", type=Path)
    parser.add_argument("--answers-file", type=Path)
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument("--stage-index", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--prefix-len", type=int, default=20)
    parser.add_argument("--text-tower", type=Path)
    parser.add_argument("--mm-projector", type=Path)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_json_or_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        prefix = handle.read(1)
        handle.seek(0)
        if prefix == "[":
            return json.load(handle)
        return [json.loads(line) for line in handle if line.strip()]


def load_jobs(args):
    if args.jobs_file:
        jobs = read_json_or_jsonl(args.jobs_file)
    elif args.question_file and args.answers_file:
        jobs = [
            {
                "question_file": str(args.question_file),
                "answers_file": str(args.answers_file),
            }
        ]
    else:
        raise ValueError("Provide --jobs-file or both --question-file and --answers-file")
    required = ("question_file", "answers_file")
    for job in jobs:
        missing = [key for key in required if not job.get(key)]
        if missing:
            raise ValueError("Malformed inference job; missing {}".format(missing))
    return jobs


def select_chunk(rows, num_chunks, chunk_idx):
    if num_chunks <= 0:
        raise ValueError("--num-chunks must be positive")
    if chunk_idx < 0 or chunk_idx >= num_chunks:
        raise ValueError("--chunk-idx must be in [0, num_chunks)")
    chunk_size = int(math.ceil(len(rows) / float(num_chunks))) if rows else 0
    if not chunk_size:
        return []
    start = chunk_idx * chunk_size
    return rows[start : start + chunk_size]


def import_method_llava(method_root):
    root = str(method_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    for name in tuple(sys.modules):
        if name == "llava" or name.startswith("llava."):
            del sys.modules[name]

    constants = importlib.import_module("llava.constants")
    conversation = importlib.import_module("llava.conversation")
    builder = importlib.import_module("llava.model.builder")
    mm_utils = importlib.import_module("llava.mm_utils")
    utils = importlib.import_module("llava.utils")
    return constants, conversation, builder, mm_utils, utils


def load_method_model(args):
    constants, conversation, builder, mm_utils, utils = import_method_llava(args.method_root)
    utils.disable_torch_init()

    adapter_method = args.method not in {"ModalPrompt", "RegLoRA", "KeepLoRA"}
    model_name = args.model_name
    if not model_name:
        model_name = "llava-lora-coinpp" if adapter_method else "llava-coinpp"

    model_path = str(args.model_path.resolve())
    model_base = str(args.model_base.resolve()) if args.model_base else None
    common = {"device": args.device}

    if args.method == "ModalPrompt":
        if not args.text_tower:
            raise ValueError("ModalPrompt requires --text-tower")
        tokenizer, model, image_processor, context_len = builder.load_pretrained_model(
            model_path,
            model_base,
            model_name,
            args.prefix_len,
            args.stage_index,
            str(args.text_tower.resolve()),
            args.num_tasks,
            mm_projector_path=(
                str(args.mm_projector.resolve()) if args.mm_projector else None
            ),
            **common
        )
    elif args.method == "OLoRA":
        tokenizer, model, image_processor, context_len = builder.load_pretrained_model(
            model_path,
            model_base,
            model_name,
            cur_task=max(0, args.stage_index - 1),
            **common
        )
    elif args.method in {"HiDe", "DISCO"}:
        if not args.text_tower:
            raise ValueError("{} requires --text-tower".format(args.method))
        tokenizer, model, image_processor, context_len = builder.load_pretrained_model(
            model_path,
            model_base,
            model_name,
            num_task=args.num_tasks,
            text_tower=str(args.text_tower.resolve()),
            **common
        )
    else:
        tokenizer, model, image_processor, context_len = builder.load_pretrained_model(
            model_path,
            model_base,
            model_name,
            **common
        )

    if args.method == "SMoLoRA":
        ins_type = max(0, args.stage_index - 1)
        changed = 0
        for module in model.modules():
            if module.__class__.__name__ == "SMoLoraLinear":
                module.ins_type = ins_type
                changed += 1
        print("SMoLoRA inference route={} modules={}".format(ins_type, changed))

    model.eval()
    return {
        "constants": constants,
        "conversation": conversation,
        "mm_utils": mm_utils,
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "context_len": context_len,
        "model_name": model_name,
    }


def first_model_device(model):
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("The loaded model has no materialized parameters")


def clean_question(text):
    text = IMAGE_TOKEN_RE.sub("", str(text or ""))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def image_prompt(text, model, constants):
    if getattr(model.config, "mm_use_im_start_end", False):
        token = (
            constants.DEFAULT_IM_START_TOKEN
            + constants.DEFAULT_IMAGE_TOKEN
            + constants.DEFAULT_IM_END_TOKEN
        )
    else:
        token = constants.DEFAULT_IMAGE_TOKEN
    return token + "\n" + text


def move_image_tensor(value, device, torch):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if isinstance(value, list):
        return [item.to(device=device, dtype=dtype, non_blocking=True) for item in value]
    return value.to(device=device, dtype=dtype, non_blocking=True)


def prepare_image(image_path, image_processor, model_config, mm_utils, device, torch):
    from PIL import Image

    with Image.open(str(image_path)) as handle:
        image = handle.convert("RGB")
        image_size = image.size
        if hasattr(mm_utils, "process_images"):
            tensor = mm_utils.process_images([image], image_processor, model_config)
        else:
            tensor = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ]
    return move_image_tensor(tensor, device, torch), image_size


def generate_one(row, args, state):
    import torch

    constants = state["constants"]
    conversation = state["conversation"]
    mm_utils = state["mm_utils"]
    tokenizer = state["tokenizer"]
    model = state["model"]
    image_processor = state["image_processor"]
    device = first_model_device(model)

    question = clean_question(row.get("text") or row.get("question"))
    image_value = row.get("image")
    has_image = bool(image_value)
    image_tensor = None
    image_size = None
    if has_image:
        image_path = args.image_folder / str(image_value)
        if not image_path.is_file():
            raise FileNotFoundError(
                "Image not found for {}: {}".format(
                    row.get("question_id") or row.get("id"), image_path
                )
            )
        prompt_question = image_prompt(question, model, constants)
        image_tensor, image_size = prepare_image(
            image_path,
            image_processor,
            model.config,
            mm_utils,
            device,
            torch,
        )
    else:
        prompt_question = question

    conv = conversation.conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], prompt_question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if has_image:
        input_ids = mm_utils.tokenizer_image_token(
            prompt,
            tokenizer,
            constants.IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)

    stop_str = (
        conv.sep
        if conv.sep_style != conversation.SeparatorStyle.TWO
        else conv.sep2
    )
    stopping = mm_utils.KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    generate_kwargs = {
        "do_sample": args.temperature > 0,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "stopping_criteria": [stopping],
    }
    if args.temperature > 0:
        generate_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        generate_kwargs["top_p"] = args.top_p
    if has_image:
        generate_kwargs["images"] = image_tensor
        generate_kwargs["image_sizes"] = [image_size]

    with torch.inference_mode():
        try:
            output_ids = model.generate(input_ids, **generate_kwargs)
        except TypeError as error:
            if "image_sizes" not in str(error):
                raise
            generate_kwargs.pop("image_sizes", None)
            output_ids = model.generate(input_ids, **generate_kwargs)

    input_length = input_ids.shape[1]
    if output_ids.shape[1] >= input_length:
        new_tokens = output_ids[:, input_length:]
    else:
        new_tokens = output_ids
    prediction = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    if stop_str and prediction.endswith(stop_str):
        prediction = prediction[: -len(stop_str)].strip()

    sample_id = str(row.get("question_id") or row.get("id"))
    metadata = dict(row.get("metadata") or {})
    dataset = row.get("dataset") or metadata.get("dataset")
    references = row.get("answers") or [row.get("answer")]
    references = [str(value) for value in references if value is not None]
    ground_truth = str(row.get("answer") or (references[0] if references else ""))
    return {
        "id": sample_id,
        "question_id": sample_id,
        "question": question,
        "prompt": question,
        "pred": prediction,
        "text": prediction,
        "gt": ground_truth,
        "answers": references,
        "dataset": dataset,
        "image": image_value,
        "has_image": has_image,
        "factor": row.get("factor"),
        "category": row.get("category"),
        "profile": row.get("profile"),
        "stage_index": args.stage_index,
        "model_id": state["model_name"],
        "answer_id": uuid.uuid4().hex,
        "metadata": metadata,
    }


def run_job(job, args, state):
    question_file = Path(job["question_file"]).resolve()
    answers_file = Path(job["answers_file"]).resolve()
    rows = read_json_or_jsonl(question_file)
    rows = select_chunk(rows, args.num_chunks, args.chunk_idx)
    if args.limit is not None:
        rows = rows[: args.limit]

    answers_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = answers_file.with_suffix(answers_file.suffix + ".tmp")
    print(
        "Inference: category={} samples={} output={}".format(
            job.get("category", "unknown"), len(rows), answers_file
        )
    )
    with temporary.open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows, start=1):
            prediction = generate_one(row, args, state)
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            if index % 25 == 0 or index == len(rows):
                output.flush()
                print("  {}/{}".format(index, len(rows)), flush=True)
    temporary.replace(answers_file)


def main():
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if not args.model_path.is_dir():
        raise FileNotFoundError("Model checkpoint not found: {}".format(args.model_path))
    if args.model_base and not args.model_base.is_dir():
        raise FileNotFoundError("Base model not found: {}".format(args.model_base))
    if not args.image_folder.is_dir():
        raise FileNotFoundError("Image folder not found: {}".format(args.image_folder))

    jobs = load_jobs(args)
    print(
        "Load CoIN++ model: method={} stage={} jobs={}".format(
            args.method, args.stage_index, len(jobs)
        )
    )
    state = load_method_model(args)
    for job in jobs:
        run_job(job, args, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
