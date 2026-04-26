from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import importlib
import os
import sys
import types
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms


RESULT_FIELDS = [
    "sample_id",
    "status",
    "input_path",
    "protection",
    "prob_fake",
    "pred_label",
    "logit_real",
    "logit_fake",
    "feature_dim",
    "feature_norm",
    "feature_path",
]


DETECTOR_CLASSES = {
    "core": ("detectors.core_detector", "CoreDetector"),
    "effort": ("detectors.effort_detector", "EffortDetector"),
    "ffd": ("detectors.ffd_detector", "FFDDetector"),
    "ucf": ("detectors.ucf_detector", "UCFDetector"),
    "xception": ("detectors.xception_detector", "XceptionDetector"),
    "spsl": ("detectors.spsl_detector", "SpslDetector"),
    "f3net": ("detectors.f3net_detector", "F3netDetector"),
    "srm": ("detectors.srm_detector", "SRMDetector"),
}


def _bootstrap_src() -> None:
    script_dir = Path(__file__).resolve().parent
    src_root = script_dir.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_bootstrap_src()

from hybrid_router.config_io import load_yaml_config
from hybrid_router.manifest_schema import load_manifest


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Run a DeepfakeBench image detector on a hybrid manifest."
    )
    parser.add_argument("--config", required=True, help="Hybrid smoke/test YAML config.")
    parser.add_argument("--manifest", help="Optional explicit manifest path.")
    parser.add_argument("--output", required=True, help="Passive result CSV to write.")
    parser.add_argument("--detector", choices=sorted(DETECTOR_CLASSES), required=True)
    parser.add_argument(
        "--deepfakebench-root",
        default=str(root / "workspace" / "repos" / "DeepfakeBench"),
        help="Path to the DeepfakeBench checkout.",
    )
    parser.add_argument(
        "--detector-config",
        default="",
        help="Optional detector YAML. Defaults to DeepfakeBench training/config/detector/<detector>.yaml.",
    )
    parser.add_argument(
        "--pretrained",
        default=str(
            root
            / "workspace"
            / "checkpoints"
            / "deepfakebench"
            / "pretrained"
            / "pretrained"
            / "xception-b5690688.pth"
        ),
        help="ImageNet Xception backbone weights used at construction time.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="DeepfakeBench detector checkpoint. Defaults to workspace/checkpoints/deepfakebench/weights/<detector>_best.pth.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5, help="Only affects pred_label.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--save-features",
        action="store_true",
        help="Persist detector feature tensors to passive_key-derived NPZ files. Disabled by default to avoid large artifacts.",
    )
    return parser.parse_args()


def install_optional_import_stubs() -> None:
    if "torch.utils.tensorboard" not in sys.modules:
        tensorboard_stub = types.ModuleType("torch.utils.tensorboard")

        class SummaryWriter:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def add_scalar(self, *args, **kwargs) -> None:
                pass

            def close(self) -> None:
                pass

        tensorboard_stub.SummaryWriter = SummaryWriter
        sys.modules["torch.utils.tensorboard"] = tensorboard_stub


def install_deepfakebench_minimal_packages(deepfakebench_root: Path) -> None:
    training_root = deepfakebench_root / "training"
    if str(training_root) not in sys.path:
        sys.path.insert(0, str(training_root))
    if str(deepfakebench_root) not in sys.path:
        sys.path.insert(0, str(deepfakebench_root))

    from metrics.registry import BACKBONE, DETECTOR, LOSSFUNC

    networks_pkg = types.ModuleType("networks")
    networks_pkg.__path__ = [str(training_root / "networks")]
    networks_pkg.BACKBONE = BACKBONE
    sys.modules["networks"] = networks_pkg

    loss_pkg = types.ModuleType("loss")
    loss_pkg.__path__ = [str(training_root / "loss")]
    loss_pkg.LOSSFUNC = LOSSFUNC
    sys.modules["loss"] = loss_pkg

    detectors_pkg = types.ModuleType("detectors")
    detectors_pkg.__path__ = [str(training_root / "detectors")]
    detectors_pkg.DETECTOR = DETECTOR
    sys.modules["detectors"] = detectors_pkg

    import networks.xception  # noqa: F401
    import loss.cross_entropy_loss  # noqa: F401
    import loss.am_softmax  # noqa: F401
    import loss.consistency_loss  # noqa: F401
    import loss.contrastive_regularization  # noqa: F401
    import loss.l1_loss  # noqa: F401


@contextmanager
def temporary_working_directory(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def load_detector_model(
    detector_name: str,
    deepfakebench_root: Path,
    detector_config_path: Path,
    pretrained_path: Path,
    checkpoint_path: Path,
    device: str,
) -> torch.nn.Module:
    install_optional_import_stubs()
    install_deepfakebench_minimal_packages(deepfakebench_root)

    module_name, class_name = DETECTOR_CLASSES[detector_name]
    detector_module = importlib.import_module(module_name)
    detector_class = getattr(detector_module, class_name)

    with detector_config_path.open("r", encoding="utf-8") as handle:
        detector_config = yaml.safe_load(handle)
    detector_config["pretrained"] = str(pretrained_path)
    detector_config["cuda"] = device.startswith("cuda")
    detector_config["cudnn"] = device.startswith("cuda")
    if detector_name == "effort":
        detector_config["clip_model_path"] = "openai/clip-vit-large-patch14"
        detector_config["clip_cache_dir"] = str(project_root() / "workspace" / "checkpoints" / "huggingface")

    with temporary_working_directory(deepfakebench_root):
        model = detector_class(detector_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_keys = list(model.state_dict())
    state_keys = list(state_dict)
    if state_keys and state_keys[0].startswith("module.") and model_keys and not model_keys[0].startswith("module."):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def make_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def resolve_input(row: Dict[str, str]) -> Path:
    if row["protection"] == "protected":
        return Path(row["planned_output_path"])
    return Path(row["source_path"])


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def empty_result(row: Dict[str, str], input_path: Path, status: str) -> Dict[str, str]:
    return {
        "sample_id": row["sample_id"],
        "status": status,
        "input_path": str(input_path),
        "protection": row["protection"],
        "prob_fake": "",
        "pred_label": "",
        "logit_real": "",
        "logit_fake": "",
        "feature_dim": "",
        "feature_norm": "",
        "feature_path": "",
    }


def feature_output_path(row: Dict[str, str], detector_name: str) -> Path | None:
    passive_key = row.get("passive_key", "")
    if not passive_key or not passive_key.lower().endswith(".npz"):
        return None
    original = Path(passive_key)
    return original.with_name(f"{original.stem}_{detector_name}{original.suffix}")


def flatten_feature(feature: torch.Tensor) -> np.ndarray:
    if feature.ndim <= 2:
        return feature.detach().cpu().numpy().astype(np.float32)
    pooled = torch.nn.functional.adaptive_avg_pool2d(feature, (1, 1)).flatten(1)
    return pooled.detach().cpu().numpy().astype(np.float32)


def flush_batch(
    detector_name: str,
    model: torch.nn.Module,
    batch_rows: List[Dict[str, str]],
    batch_tensors: List[torch.Tensor],
    batch_paths: List[Path],
    output_rows: List[Dict[str, str]],
    device: str,
    threshold: float,
    save_features: bool,
) -> None:
    if not batch_rows:
        return
    images = torch.stack(batch_tensors, dim=0).to(device)
    labels = torch.tensor([1 if row["label"] == "fake" else 0 for row in batch_rows], dtype=torch.long, device=device)
    predictions = model({"image": images, "label": labels}, inference=True)
    logits = predictions["cls"]
    probs = torch.softmax(logits, dim=1)[:, 1]
    features = flatten_feature(predictions["feat"])

    for row, input_path, logit, feature_vector, prob in zip(batch_rows, batch_paths, logits, features, probs):
        prob_fake = float(prob.detach().cpu().item())
        pred_label = 1 if prob_fake >= threshold else 0
        feature_path_text = ""
        if save_features:
            feature_path = feature_output_path(row, detector_name)
            if feature_path is not None:
                feature_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    feature_path,
                    sample_id=row["sample_id"],
                    input_path=str(input_path),
                    arch=f"deepfakebench_{detector_name}",
                    prob_fake=prob_fake,
                    pred_label=pred_label,
                    feature=feature_vector,
                )
                feature_path_text = str(feature_path)
        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "status": "ok",
                "input_path": str(input_path),
                "protection": row["protection"],
                "prob_fake": f"{prob_fake:.6f}",
                "pred_label": str(pred_label),
                "logit_real": f"{float(logit[0].detach().cpu().item()):.6f}",
                "logit_fake": f"{float(logit[1].detach().cpu().item()):.6f}",
                "feature_dim": str(int(feature_vector.shape[0])),
                "feature_norm": f"{float(np.linalg.norm(feature_vector)):.6f}",
                "feature_path": feature_path_text,
            }
        )


def main() -> None:
    args = parse_args()
    config = load_yaml_config(Path(args.config))
    manifest_path = Path(args.manifest or config["paths"]["manifest_path"])
    deepfakebench_root = Path(args.deepfakebench_root)
    detector_config_path = (
        Path(args.detector_config)
        if args.detector_config
        else deepfakebench_root / "training" / "config" / "detector" / f"{args.detector}.yaml"
    )
    pretrained_path = Path(args.pretrained)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else project_root() / "workspace" / "checkpoints" / "deepfakebench" / "weights" / f"{args.detector}_best.pth"
    )

    for required in [manifest_path, detector_config_path, pretrained_path, checkpoint_path]:
        if not required.exists():
            raise FileNotFoundError(required)

    model = load_detector_model(
        detector_name=args.detector,
        deepfakebench_root=deepfakebench_root,
        detector_config_path=detector_config_path,
        pretrained_path=pretrained_path,
        checkpoint_path=checkpoint_path,
        device=args.device,
    )
    transform = make_transform(args.image_size)
    manifest_rows = load_manifest(manifest_path)
    budget = args.limit if args.limit > 0 else None

    output_rows: List[Dict[str, str]] = []
    batch_rows: List[Dict[str, str]] = []
    batch_tensors: List[torch.Tensor] = []
    batch_paths: List[Path] = []

    with torch.no_grad():
        for index, row in enumerate(manifest_rows):
            input_path = resolve_input(row)
            if budget is not None and index >= budget:
                output_rows.append(empty_result(row, input_path, "skipped_limit"))
                continue
            if not input_path.exists():
                output_rows.append(empty_result(row, input_path, "missing_input"))
                continue

            image = Image.open(input_path).convert("RGB")
            batch_rows.append(row)
            batch_tensors.append(transform(image))
            batch_paths.append(input_path)

            if len(batch_rows) >= args.batch_size:
                flush_batch(
                    args.detector,
                    model,
                    batch_rows,
                    batch_tensors,
                    batch_paths,
                    output_rows,
                    args.device,
                    args.threshold,
                    args.save_features,
                )
                batch_rows, batch_tensors, batch_paths = [], [], []

        flush_batch(
            args.detector,
            model,
            batch_rows,
            batch_tensors,
            batch_paths,
            output_rows,
            args.device,
            args.threshold,
            args.save_features,
        )

    output_path = Path(args.output)
    write_csv(output_path, output_rows)
    print(f"Wrote DeepfakeBench {args.detector} passive results to {output_path}")


if __name__ == "__main__":
    main()
