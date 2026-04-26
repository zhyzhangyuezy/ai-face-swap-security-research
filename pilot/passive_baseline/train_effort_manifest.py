from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Sampler
from transformers import CLIPVisionModel

from train_clip_adapter import (
    CLIP_MEAN,
    CLIP_STD,
    ManifestFrameDataset,
    append_results,
    collect_probabilities,
    evaluate,
    load_manifest_records,
    parse_eval_manifest,
    project_root,
    resolve_path,
    select_threshold,
    split_train_records,
)


class SVDResidualLinear(nn.Module):
    def __init__(self, base: nn.Linear, residual_rank: int) -> None:
        super().__init__()
        weight = base.weight.detach().float()
        out_features, in_features = weight.shape
        min_dim = min(out_features, in_features)
        keep_rank = max(0, min(min_dim, min_dim - residual_rank))
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        main = u[:, :keep_rank] @ torch.diag(s[:keep_rank]) @ vh[:keep_rank, :]

        self.weight_main = nn.Parameter(main.to(base.weight.dtype), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

        if keep_rank < min_dim:
            self.u_residual = nn.Parameter(u[:, keep_rank:].to(base.weight.dtype))
            self.s_residual = nn.Parameter(s[keep_rank:].to(base.weight.dtype))
            self.vh_residual = nn.Parameter(vh[keep_rank:, :].to(base.weight.dtype))
        else:
            self.register_parameter("u_residual", None)
            self.register_parameter("s_residual", None)
            self.register_parameter("vh_residual", None)

    def current_weight(self) -> torch.Tensor:
        if self.s_residual is None:
            return self.weight_main
        residual = self.u_residual @ torch.diag(self.s_residual) @ self.vh_residual
        return self.weight_main + residual

    def orthogonal_loss(self) -> torch.Tensor:
        if self.s_residual is None:
            return self.weight_main.new_tensor(0.0)
        u_eye = torch.eye(self.u_residual.shape[1], device=self.u_residual.device, dtype=self.u_residual.dtype)
        v_eye = torch.eye(self.vh_residual.shape[0], device=self.vh_residual.device, dtype=self.vh_residual.dtype)
        u_loss = torch.norm(self.u_residual.transpose(0, 1) @ self.u_residual - u_eye, p="fro")
        v_loss = torch.norm(self.vh_residual @ self.vh_residual.transpose(0, 1) - v_eye, p="fro")
        return 0.5 * (u_loss + v_loss)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(values, self.current_weight(), self.bias)


def replace_attention_linears(module: nn.Module, residual_rank: int) -> int:
    replaced = 0
    for child_name, child in module.named_children():
        if "self_attn" in child_name:
            for name, submodule in list(child.named_children()):
                if isinstance(submodule, nn.Linear):
                    setattr(child, name, SVDResidualLinear(submodule, residual_rank=residual_rank))
                    replaced += 1
        else:
            replaced += replace_attention_linears(child, residual_rank)
    return replaced


class CosineClassifier(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, scale_init: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.empty(2, hidden_dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(max(1e-3, scale_init))))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.dropout(self.norm(features))
        features = F.normalize(features, dim=-1)
        weights = F.normalize(self.weight, dim=-1)
        return self.logit_scale.exp().clamp(max=100.0) * (features @ weights.t())


class IndependentDualClassifier(nn.Module):
    """Real-concept realness and artifact evidence as independent logits."""

    def __init__(self, hidden_dim: int, dropout: float, prototype_count: int, scale_init: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.real_prototypes = nn.Parameter(torch.empty(max(1, prototype_count), hidden_dim))
        self.fake_weight = nn.Parameter(torch.empty(hidden_dim))
        self.fake_bias = nn.Parameter(torch.zeros(()))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(max(1e-3, scale_init))))
        nn.init.xavier_uniform_(self.real_prototypes)
        nn.init.normal_(self.fake_weight, std=hidden_dim**-0.5)

    def normalized_features(self, features: torch.Tensor, apply_dropout: bool = True) -> torch.Tensor:
        values = self.norm(features)
        if apply_dropout:
            values = self.dropout(values)
        return F.normalize(values, dim=-1)

    def normalized_real_prototypes(self) -> torch.Tensor:
        return F.normalize(self.real_prototypes, dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.normalized_features(features)
        scale = self.logit_scale.exp().clamp(max=100.0)
        real_scores = normalized @ self.normalized_real_prototypes().t()
        real_logit = real_scores.max(dim=1).values * scale
        fake_logit = (normalized @ F.normalize(self.fake_weight, dim=0)) * scale + self.fake_bias
        return torch.stack([real_logit, fake_logit], dim=1)


class EffortStyleDetector(nn.Module):
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        residual_rank: int,
        dropout: float,
        pooling: str = "cls",
        attn_hidden: int = 256,
        mean_gate_init: float = 0.2,
        use_patch_mil: bool = False,
        head_type: str = "linear",
        cosine_scale_init: float = 16.0,
        pool_topk_frac: float = 0.25,
        real_concept_prototypes: int = 16,
        real_concept_scale_init: float = 16.0,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.use_patch_mil = use_patch_mil
        self.head_type = head_type
        self.pool_topk_frac = pool_topk_frac
        self.vision = CLIPVisionModel.from_pretrained(model_name, cache_dir=cache_dir)
        for parameter in self.vision.parameters():
            parameter.requires_grad = False
        self.replaced_linears = replace_attention_linears(self.vision.vision_model, residual_rank=residual_rank)
        hidden_dim = int(self.vision.config.hidden_size)
        if pooling in {"patch_attn", "patch_attn_mean", "patch_attn_topk_mean"}:
            self.attn_pool = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, attn_hidden),
                nn.Tanh(),
                nn.Linear(attn_hidden, 1),
            )
            if pooling in {"patch_attn_mean", "patch_attn_topk_mean"}:
                clipped_gate = min(0.99, max(0.01, mean_gate_init))
                self.mean_gate_logit = nn.Parameter(torch.tensor(math.log(clipped_gate / (1.0 - clipped_gate))))
        elif pooling not in {"cls", "patch_mean"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        if head_type == "linear":
            self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))
        elif head_type == "cosine":
            self.head = CosineClassifier(hidden_dim, dropout, cosine_scale_init)
        elif head_type == "independent_dual":
            self.head = IndependentDualClassifier(
                hidden_dim,
                dropout,
                prototype_count=real_concept_prototypes,
                scale_init=real_concept_scale_init,
            )
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")
        if use_patch_mil:
            self.patch_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(
        self,
        images: torch.Tensor,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        outputs = self.vision(pixel_values=images)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        if self.pooling == "cls":
            pooled = outputs.pooler_output
        else:
            if self.pooling == "patch_mean":
                pooled = patch_tokens.mean(dim=1)
            else:
                attn_logits = self.attn_pool(patch_tokens).squeeze(-1)
                weights = torch.softmax(attn_logits, dim=1)
                attended = torch.sum(patch_tokens * weights.unsqueeze(-1), dim=1)
                if self.pooling == "patch_attn_mean":
                    patch_mean = patch_tokens.mean(dim=1)
                    mean_gate = torch.sigmoid(self.mean_gate_logit)
                    pooled = attended + mean_gate * (patch_mean - attended)
                elif self.pooling == "patch_attn_topk_mean":
                    num_patches = patch_tokens.shape[1]
                    topk = max(1, min(num_patches, int(math.ceil(num_patches * self.pool_topk_frac))))
                    topk_index = torch.topk(attn_logits, k=topk, dim=1).indices.unsqueeze(-1)
                    topk_index = topk_index.expand(-1, -1, patch_tokens.shape[-1])
                    topk_mean = torch.gather(patch_tokens, dim=1, index=topk_index).mean(dim=1)
                    mean_gate = torch.sigmoid(self.mean_gate_logit)
                    pooled = attended + mean_gate * (topk_mean - attended)
                else:
                    pooled = attended
        logits = self.head(pooled)
        if return_aux:
            aux = {"patch_tokens": patch_tokens}
            if self.use_patch_mil:
                aux["patch_logits"] = self.patch_head(patch_tokens).squeeze(-1)
            return logits, pooled, aux
        return logits, pooled

    def orthogonal_loss(self) -> torch.Tensor:
        losses = [module.orthogonal_loss() for module in self.modules() if isinstance(module, SVDResidualLinear)]
        if not losses:
            return next(self.parameters()).new_tensor(0.0)
        return torch.stack(losses).mean()


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train an Effort-style CLIP SVD residual detector from manifests.")
    parser.add_argument("--train-manifest", action="append", required=True)
    parser.add_argument("--eval-manifest", action="append", default=[], help="NAME=manifest.csv. Repeat for cross eval.")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--model-name", default="openai/clip-vit-large-patch14")
    parser.add_argument("--cache-dir", default=str(root / "workspace" / "checkpoints" / "huggingface"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--residual-rank", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--pooling",
        choices=["cls", "patch_mean", "patch_attn", "patch_attn_mean", "patch_attn_topk_mean"],
        default="cls",
    )
    parser.add_argument("--attn-hidden", type=int, default=256)
    parser.add_argument("--mean-gate-init", type=float, default=0.2)
    parser.add_argument("--pool-topk-frac", type=float, default=0.25)
    parser.add_argument("--frequency-emphasis", type=float, default=0.0)
    parser.add_argument("--orthogonal-weight", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-type", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--real-class-weight", type=float, default=1.0)
    parser.add_argument("--fake-class-weight", type=float, default=1.0)
    parser.add_argument("--fake-margin-weight", type=float, default=0.0)
    parser.add_argument("--fake-margin-logit", type=float, default=0.0)
    parser.add_argument("--real-margin-weight", type=float, default=0.0)
    parser.add_argument("--real-margin-logit", type=float, default=0.0)
    parser.add_argument("--patch-mil-weight", type=float, default=0.0)
    parser.add_argument("--patch-mil-real-weight", type=float, default=1.0)
    parser.add_argument("--patch-mil-topk-frac", type=float, default=0.15)
    parser.add_argument(
        "--patch-mil-manifest-token",
        action="append",
        default=[],
        help=(
            "Restrict patch-MIL supervision to train rows whose source manifest path contains "
            "one of these substrings. Repeat for multiple tokens."
        ),
    )
    parser.add_argument("--protected-real-patch-guard-weight", type=float, default=0.0)
    parser.add_argument("--protected-real-patch-guard-topk-frac", type=float, default=0.15)
    parser.add_argument(
        "--protected-real-patch-guard-manifest-token",
        action="append",
        default=[],
        help=(
            "Restrict protected-real patch guard to train rows whose source manifest path contains "
            "one of these substrings. Repeat for multiple tokens."
        ),
    )
    parser.add_argument(
        "--protected-real-patch-guard-note-token",
        action="append",
        default=[],
        help=(
            "Restrict protected-real patch guard to train rows whose manifest notes contain "
            "one of these substrings. Repeat for multiple tokens."
        ),
    )
    parser.add_argument("--pair-consistency-weight", type=float, default=0.0)
    parser.add_argument("--pair-consistency-margin", type=float, default=0.15)
    parser.add_argument("--pair-consistency-hard-fake-prob-max", type=float, default=1.0)
    parser.add_argument("--source-family-rank-weight", type=float, default=0.0)
    parser.add_argument("--source-family-rank-margin", type=float, default=0.05)
    parser.add_argument("--source-family-rank-topk-frac", type=float, default=0.10)
    parser.add_argument("--source-family-rank-hard-fake-prob-max", type=float, default=1.0)
    parser.add_argument(
        "--source-family-rank-manifest-token",
        action="append",
        default=[],
        help=(
            "Restrict source-family ranking to fake train rows whose source manifest path contains "
            "one of these substrings. Repeat for multiple tokens."
        ),
    )
    parser.add_argument(
        "--source-family-rank-note-token",
        action="append",
        default=[],
        help=(
            "Restrict source-family ranking to fake train rows whose manifest notes contain "
            "one of these substrings. Repeat for multiple tokens."
        ),
    )
    parser.add_argument("--source-family-rank-extra-weight", type=float, default=0.0)
    parser.add_argument(
        "--source-family-rank-extra-manifest-token",
        action="append",
        default=[],
        help=(
            "Add an extra targeted source-family ranking term on fake train rows whose source manifest "
            "path contains one of these substrings. Repeat for multiple tokens."
        ),
    )
    parser.add_argument(
        "--source-family-rank-extra-note-token",
        action="append",
        default=[],
        help=(
            "Add an extra targeted source-family ranking term on fake train rows whose manifest notes "
            "contain one of these substrings. Repeat for multiple tokens."
        ),
    )
    parser.add_argument("--source-family-real-guard-weight", type=float, default=0.0)
    parser.add_argument("--source-family-real-guard-min-sim", type=float, default=0.92)
    parser.add_argument("--source-family-real-guard-topk-frac", type=float, default=0.10)
    parser.add_argument(
        "--source-family-real-guard-manifest-token",
        action="append",
        default=[],
        help=(
            "Restrict source-family real guard to train rows whose source manifest path contains "
            "one of these substrings. Repeat for multiple tokens."
        ),
    )
    parser.add_argument(
        "--source-family-real-guard-note-token",
        action="append",
        default=[],
        help=(
            "Restrict source-family real guard to train rows whose manifest notes contain "
            "one of these substrings. Repeat for multiple tokens."
        ),
    )
    parser.add_argument(
        "--pair-group-prefix",
        action="append",
        default=[],
        help=(
            "Restrict pair/source-family auxiliary losses to source_group prefixes. "
            "Repeat for multiple prefixes, e.g. --pair-group-prefix uadfv- --pair-group-prefix ffpp-."
        ),
    )
    parser.add_argument("--head-type", choices=["linear", "cosine", "independent_dual"], default="linear")
    parser.add_argument("--cosine-scale-init", type=float, default=16.0)
    parser.add_argument("--real-concept-prototypes", type=int, default=16)
    parser.add_argument("--real-concept-scale-init", type=float, default=16.0)
    parser.add_argument("--real-concept-weight", type=float, default=0.0)
    parser.add_argument("--real-concept-real-min-sim", type=float, default=0.35)
    parser.add_argument("--real-concept-fake-max-sim", type=float, default=0.05)
    parser.add_argument("--real-concept-fake-weight", type=float, default=1.0)
    parser.add_argument(
        "--real-concept-use-sample-weights",
        action="store_true",
        help=(
            "Apply manifest sample weights to the real-concept auxiliary loss. "
            "By default the concept boundary is unweighted so low-weight domain calibration rows can still shape it."
        ),
    )
    parser.add_argument("--targeted-fake-margin-weight", type=float, default=0.0)
    parser.add_argument("--targeted-fake-margin-logit", type=float, default=2.5)
    parser.add_argument(
        "--targeted-fake-margin-manifest-token",
        action="append",
        default=[],
        help=(
            "Apply an unweighted hard fake logit-margin term to fake train rows whose source manifest path contains "
            "one of these substrings. Repeat for multiple tokens."
        ),
    )
    parser.add_argument(
        "--targeted-fake-margin-note-token",
        action="append",
        default=[],
        help=(
            "Apply an unweighted hard fake logit-margin term to fake train rows whose manifest notes contain "
            "one of these substrings. Repeat for multiple tokens."
        ),
    )
    parser.add_argument("--targeted-real-margin-weight", type=float, default=0.0)
    parser.add_argument("--targeted-real-margin-logit", type=float, default=2.5)
    parser.add_argument(
        "--targeted-real-margin-manifest-token",
        action="append",
        default=[],
        help=(
            "Apply an unweighted hard real logit-margin term to protected-real train rows whose source manifest "
            "path contains one of these substrings. Repeat for multiple tokens."
        ),
    )
    parser.add_argument(
        "--targeted-real-margin-note-token",
        action="append",
        default=[],
        help=(
            "Apply an unweighted hard real logit-margin term to protected-real train rows whose manifest notes "
            "contain one of these substrings. Repeat for multiple tokens."
        ),
    )
    parser.add_argument(
        "--targeted-real-margin-note-token-weight",
        action="append",
        default=[],
        help=(
            "Apply a weighted hard real logit-margin term to protected-real train rows whose notes contain "
            "TOKEN. Format TOKEN=WEIGHT; repeat for multiple domain/slice weights."
        ),
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(root / "pilot" / "passive_baseline" / "runs"))
    parser.add_argument("--results-tsv", default=str(root / "pilot" / "passive_baseline" / "effort_results.tsv"))
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--include-protected-real-train", action="store_true")
    parser.add_argument(
        "--protected-real-aux-only",
        action="store_true",
        help=(
            "When protected real rows are included in training, keep them available for "
            "pair/source-family auxiliary losses but exclude them from the main CE classifier loss."
        ),
    )
    parser.add_argument(
        "--pair-ignore-protected-real",
        action="store_true",
        help=(
            "When protected real rows are included in training, keep them available as singleton "
            "auxiliary rows but exclude them from paired-group construction to avoid duplicating fake rows."
        ),
    )
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_transforms(image_size: int, frequency_emphasis: float = 0.0) -> tuple[nn.Module, nn.Module]:
    from torchvision import transforms

    normalize = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    tensor_steps: list[nn.Module] = [transforms.ToTensor()]
    if frequency_emphasis > 0:
        tensor_steps.append(HighFrequencyEmphasis(frequency_emphasis))
    tensor_steps.append(normalize)
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.90, 1.10)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.01),
            *tensor_steps,
        ]
    )
    eval_tf = transforms.Compose([transforms.Resize((image_size, image_size)), *tensor_steps])
    return train_tf, eval_tf


def make_loader(records, transform, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ManifestFrameDataset(records, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def record_token_weight(
    record,
    manifest_tokens: list[str] | None = None,
    note_tokens: list[str] | None = None,
) -> float:
    manifest_values = [token.lower() for token in (manifest_tokens or []) if token]
    note_values = [token.lower() for token in (note_tokens or []) if token]
    if not manifest_values and not note_values:
        return 1.0
    source_manifest = str(getattr(record, "source_manifest", "")).lower()
    notes = str(getattr(record, "notes", "")).lower()
    manifest_match = not manifest_values or any(token in source_manifest for token in manifest_values)
    note_match = not note_values or any(token in notes for token in note_values)
    return 1.0 if manifest_match and note_match else 0.0


def parse_weighted_note_token_specs(values: list[str] | None) -> list[tuple[str, float]]:
    specs: list[tuple[str, float]] = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Weighted note token must use TOKEN=WEIGHT format: {value!r}")
        token, weight_text = value.rsplit("=", 1)
        token = token.strip().lower()
        if not token:
            raise ValueError(f"Weighted note token cannot be empty: {value!r}")
        try:
            weight = float(weight_text)
        except ValueError as exc:
            raise ValueError(f"Invalid weighted note-token weight in {value!r}") from exc
        specs.append((token, max(0.0, weight)))
    return specs


def weighted_note_token_weight(record, specs: list[tuple[str, float]] | None = None) -> float:
    if not specs:
        return 0.0
    notes = str(getattr(record, "notes", "")).lower()
    matched_weights = [weight for token, weight in specs if token in notes]
    return max(matched_weights, default=0.0)


def patch_mil_record_weight(
    record,
    manifest_tokens: list[str] | None = None,
    note_tokens: list[str] | None = None,
) -> float:
    return record_token_weight(record, manifest_tokens=manifest_tokens, note_tokens=note_tokens)


class EffortTrainManifestFrameDataset(ManifestFrameDataset):
    def __init__(
        self,
        records,
        transform,
        group_to_index: dict[str, int],
        protected_real_aux_only: bool = False,
        patch_mil_manifest_tokens: list[str] | None = None,
        protected_real_patch_guard_manifest_tokens: list[str] | None = None,
        protected_real_patch_guard_note_tokens: list[str] | None = None,
        source_family_rank_manifest_tokens: list[str] | None = None,
        source_family_rank_note_tokens: list[str] | None = None,
        source_family_rank_extra_manifest_tokens: list[str] | None = None,
        source_family_rank_extra_note_tokens: list[str] | None = None,
        source_family_real_guard_manifest_tokens: list[str] | None = None,
        source_family_real_guard_note_tokens: list[str] | None = None,
        targeted_fake_margin_manifest_tokens: list[str] | None = None,
        targeted_fake_margin_note_tokens: list[str] | None = None,
        targeted_real_margin_manifest_tokens: list[str] | None = None,
        targeted_real_margin_note_tokens: list[str] | None = None,
        targeted_real_margin_note_token_weights: list[str] | None = None,
    ) -> None:
        super().__init__(records, transform)
        self.group_to_index = group_to_index
        self.protected_real_aux_only = protected_real_aux_only
        self.patch_mil_manifest_tokens = patch_mil_manifest_tokens or []
        self.protected_real_patch_guard_manifest_tokens = protected_real_patch_guard_manifest_tokens or []
        self.protected_real_patch_guard_note_tokens = protected_real_patch_guard_note_tokens or []
        self.source_family_rank_manifest_tokens = source_family_rank_manifest_tokens or []
        self.source_family_rank_note_tokens = source_family_rank_note_tokens or []
        self.source_family_rank_extra_manifest_tokens = source_family_rank_extra_manifest_tokens or []
        self.source_family_rank_extra_note_tokens = source_family_rank_extra_note_tokens or []
        self.source_family_real_guard_manifest_tokens = source_family_real_guard_manifest_tokens or []
        self.source_family_real_guard_note_tokens = source_family_real_guard_note_tokens or []
        self.targeted_fake_margin_manifest_tokens = targeted_fake_margin_manifest_tokens or []
        self.targeted_fake_margin_note_tokens = targeted_fake_margin_note_tokens or []
        self.targeted_real_margin_manifest_tokens = targeted_real_margin_manifest_tokens or []
        self.targeted_real_margin_note_tokens = targeted_real_margin_note_tokens or []
        self.targeted_real_margin_note_token_weights = parse_weighted_note_token_specs(
            targeted_real_margin_note_token_weights
        )

    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        record = self.records[index]
        group_name = getattr(record, "source_group", "")
        ce_weight = float(getattr(record, "sample_weight", 1.0))
        if self.protected_real_aux_only and record.label == 0 and record.protection == "protected":
            ce_weight = 0.0
        patch_mil_weight = patch_mil_record_weight(record, self.patch_mil_manifest_tokens)
        protected_real_flag = 0.0
        if record.label == 0 and record.protection == "protected":
            protected_real_flag = record_token_weight(
                record,
                manifest_tokens=self.protected_real_patch_guard_manifest_tokens,
                note_tokens=self.protected_real_patch_guard_note_tokens,
            )
        source_rank_flag = 0.0
        if record.label == 1:
            source_rank_flag = record_token_weight(
                record,
                manifest_tokens=self.source_family_rank_manifest_tokens,
                note_tokens=self.source_family_rank_note_tokens,
            )
        source_rank_extra_flag = 0.0
        if record.label == 1:
            source_rank_extra_flag = record_token_weight(
                record,
                manifest_tokens=self.source_family_rank_extra_manifest_tokens,
                note_tokens=self.source_family_rank_extra_note_tokens,
            )
        source_real_guard_flag = 0.0
        if record.label == 0:
            source_real_guard_flag = record_token_weight(
                record,
                manifest_tokens=self.source_family_real_guard_manifest_tokens,
                note_tokens=self.source_family_real_guard_note_tokens,
            )
        targeted_fake_margin_flag = 0.0
        if record.label == 1:
            targeted_fake_margin_flag = record_token_weight(
                record,
                manifest_tokens=self.targeted_fake_margin_manifest_tokens,
                note_tokens=self.targeted_fake_margin_note_tokens,
            )
        targeted_real_margin_flag = 0.0
        if record.label == 0 and record.protection == "protected":
            has_unweighted_target = bool(
                self.targeted_real_margin_manifest_tokens or self.targeted_real_margin_note_tokens
            )
            if has_unweighted_target or not self.targeted_real_margin_note_token_weights:
                targeted_real_margin_flag = record_token_weight(
                    record,
                    manifest_tokens=self.targeted_real_margin_manifest_tokens,
                    note_tokens=self.targeted_real_margin_note_tokens,
                )
            targeted_real_margin_flag = max(
                targeted_real_margin_flag,
                weighted_note_token_weight(record, self.targeted_real_margin_note_token_weights),
            )
        return (
            image,
            label,
            self.group_to_index.get(group_name, -1),
            ce_weight,
            patch_mil_weight,
            protected_real_flag,
            source_rank_flag,
            source_rank_extra_flag,
            source_real_guard_flag,
            targeted_fake_margin_flag,
            targeted_real_margin_flag,
        )


def build_pair_group_index(
    records,
    group_prefixes: list[str] | None = None,
    ignore_protected_real: bool = False,
) -> tuple[dict[str, int], dict[str, int]]:
    prefixes = tuple(prefix for prefix in (group_prefixes or []) if prefix)
    label_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in records:
        group_name = getattr(record, "source_group", "")
        if not group_name:
            continue
        if prefixes and not group_name.startswith(prefixes):
            continue
        if ignore_protected_real and record.label == 0 and getattr(record, "protection", "") == "protected":
            continue
        label_counts[group_name][int(record.label)] += 1
    valid_groups = sorted(group_name for group_name, counts in label_counts.items() if counts[0] > 0 and counts[1] > 0)
    valid_set = set(valid_groups)
    stats = {
        "candidate_groups": len(label_counts),
        "paired_groups": len(valid_groups),
        "pair_group_prefixes": list(prefixes),
        "pair_ignore_protected_real": bool(ignore_protected_real),
        "pairable_records": sum(1 for record in records if getattr(record, "source_group", "") in valid_set),
        "pairable_real_records": sum(
            1 for record in records if record.label == 0 and getattr(record, "source_group", "") in valid_set
        ),
        "pairable_fake_records": sum(
            1 for record in records if record.label == 1 and getattr(record, "source_group", "") in valid_set
        ),
    }
    return {group_name: index for index, group_name in enumerate(valid_groups)}, stats


class PairBatchSampler(Sampler[list[int]]):
    def __init__(self, records, group_to_index: dict[str, int], batch_size: int, seed: int) -> None:
        self.records = records
        self.group_to_index = group_to_index
        self.batch_size = max(2, int(batch_size))
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        by_group: dict[str, dict[int, list[int]]] = defaultdict(lambda: {0: [], 1: []})
        for index, record in enumerate(self.records):
            group_name = getattr(record, "source_group", "")
            if group_name in self.group_to_index:
                by_group[group_name][int(record.label)].append(index)

        units: list[list[int]] = []
        paired_indices: set[int] = set()
        for buckets in by_group.values():
            real_indices = buckets[0][:]
            fake_indices = buckets[1][:]
            rng.shuffle(real_indices)
            rng.shuffle(fake_indices)
            pair_count = max(len(real_indices), len(fake_indices))
            for offset in range(pair_count):
                pair = [real_indices[offset % len(real_indices)], fake_indices[offset % len(fake_indices)]]
                paired_indices.update(pair)
                units.append(pair)

        units.extend([[index] for index in range(len(self.records)) if index not in paired_indices])
        rng.shuffle(units)
        batch: list[int] = []
        for unit in units:
            if batch and len(batch) + len(unit) > self.batch_size:
                yield batch
                batch = []
            batch.extend(unit)
        if batch:
            yield batch

    def __len__(self) -> int:
        return math.ceil(max(1, len(self.records)) / self.batch_size)


def make_pair_loader(
    records,
    transform,
    batch_size: int,
    workers: int,
    group_to_index: dict[str, int],
    seed: int,
    protected_real_aux_only: bool = False,
    patch_mil_manifest_tokens: list[str] | None = None,
    protected_real_patch_guard_manifest_tokens: list[str] | None = None,
    protected_real_patch_guard_note_tokens: list[str] | None = None,
    source_family_rank_manifest_tokens: list[str] | None = None,
    source_family_rank_note_tokens: list[str] | None = None,
    source_family_rank_extra_manifest_tokens: list[str] | None = None,
    source_family_rank_extra_note_tokens: list[str] | None = None,
    source_family_real_guard_manifest_tokens: list[str] | None = None,
    source_family_real_guard_note_tokens: list[str] | None = None,
    targeted_fake_margin_manifest_tokens: list[str] | None = None,
    targeted_fake_margin_note_tokens: list[str] | None = None,
    targeted_real_margin_manifest_tokens: list[str] | None = None,
    targeted_real_margin_note_tokens: list[str] | None = None,
    targeted_real_margin_note_token_weights: list[str] | None = None,
) -> DataLoader:
    return DataLoader(
        EffortTrainManifestFrameDataset(
            records,
            transform,
            group_to_index,
            protected_real_aux_only,
            patch_mil_manifest_tokens,
            protected_real_patch_guard_manifest_tokens,
            protected_real_patch_guard_note_tokens,
            source_family_rank_manifest_tokens,
            source_family_rank_note_tokens,
            source_family_rank_extra_manifest_tokens,
            source_family_rank_extra_note_tokens,
            source_family_real_guard_manifest_tokens,
            source_family_real_guard_note_tokens,
            targeted_fake_margin_manifest_tokens,
            targeted_fake_margin_note_tokens,
            targeted_real_margin_manifest_tokens,
            targeted_real_margin_note_tokens,
            targeted_real_margin_note_token_weights,
        ),
        batch_sampler=PairBatchSampler(records, group_to_index, batch_size, seed),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_train_loader(
    records,
    transform,
    batch_size: int,
    workers: int,
    shuffle: bool,
    protected_real_aux_only: bool = False,
    patch_mil_manifest_tokens: list[str] | None = None,
    protected_real_patch_guard_manifest_tokens: list[str] | None = None,
    protected_real_patch_guard_note_tokens: list[str] | None = None,
    source_family_rank_manifest_tokens: list[str] | None = None,
    source_family_rank_note_tokens: list[str] | None = None,
    source_family_rank_extra_manifest_tokens: list[str] | None = None,
    source_family_rank_extra_note_tokens: list[str] | None = None,
    source_family_real_guard_manifest_tokens: list[str] | None = None,
    source_family_real_guard_note_tokens: list[str] | None = None,
    targeted_fake_margin_manifest_tokens: list[str] | None = None,
    targeted_fake_margin_note_tokens: list[str] | None = None,
    targeted_real_margin_manifest_tokens: list[str] | None = None,
    targeted_real_margin_note_tokens: list[str] | None = None,
    targeted_real_margin_note_token_weights: list[str] | None = None,
) -> DataLoader:
    return DataLoader(
        EffortTrainManifestFrameDataset(
            records,
            transform,
            group_to_index={},
            protected_real_aux_only=protected_real_aux_only,
            patch_mil_manifest_tokens=patch_mil_manifest_tokens,
            protected_real_patch_guard_manifest_tokens=protected_real_patch_guard_manifest_tokens,
            protected_real_patch_guard_note_tokens=protected_real_patch_guard_note_tokens,
            source_family_rank_manifest_tokens=source_family_rank_manifest_tokens,
            source_family_rank_note_tokens=source_family_rank_note_tokens,
            source_family_rank_extra_manifest_tokens=source_family_rank_extra_manifest_tokens,
            source_family_rank_extra_note_tokens=source_family_rank_extra_note_tokens,
            source_family_real_guard_manifest_tokens=source_family_real_guard_manifest_tokens,
            source_family_real_guard_note_tokens=source_family_real_guard_note_tokens,
            targeted_fake_margin_manifest_tokens=targeted_fake_margin_manifest_tokens,
            targeted_fake_margin_note_tokens=targeted_fake_margin_note_tokens,
            targeted_real_margin_manifest_tokens=targeted_real_margin_manifest_tokens,
            targeted_real_margin_note_tokens=targeted_real_margin_note_tokens,
            targeted_real_margin_note_token_weights=targeted_real_margin_note_token_weights,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def trainable_parameters(model: nn.Module):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def checkpoint_payload(model: EffortStyleDetector, state_dict: dict[str, torch.Tensor], args: argparse.Namespace) -> dict:
    return {
        "state_dict": state_dict,
        "model_name": args.model_name,
        "residual_rank": args.residual_rank,
        "dropout": args.dropout,
        "image_size": args.image_size,
        "pooling": args.pooling,
        "attn_hidden": args.attn_hidden,
        "mean_gate_init": args.mean_gate_init,
        "pool_topk_frac": args.pool_topk_frac,
        "frequency_emphasis": args.frequency_emphasis,
        "patch_mil_weight": args.patch_mil_weight,
        "patch_mil_real_weight": args.patch_mil_real_weight,
        "patch_mil_topk_frac": args.patch_mil_topk_frac,
        "patch_mil_manifest_token": args.patch_mil_manifest_token,
        "protected_real_patch_guard_weight": args.protected_real_patch_guard_weight,
        "protected_real_patch_guard_topk_frac": args.protected_real_patch_guard_topk_frac,
        "protected_real_patch_guard_manifest_token": args.protected_real_patch_guard_manifest_token,
        "protected_real_patch_guard_note_token": args.protected_real_patch_guard_note_token,
        "use_patch_mil": args.patch_mil_weight > 0 or args.protected_real_patch_guard_weight > 0,
        "pair_consistency_weight": args.pair_consistency_weight,
        "pair_consistency_margin": args.pair_consistency_margin,
        "pair_consistency_hard_fake_prob_max": args.pair_consistency_hard_fake_prob_max,
        "source_family_rank_weight": args.source_family_rank_weight,
        "source_family_rank_margin": args.source_family_rank_margin,
        "source_family_rank_topk_frac": args.source_family_rank_topk_frac,
        "source_family_rank_hard_fake_prob_max": args.source_family_rank_hard_fake_prob_max,
        "source_family_rank_manifest_token": args.source_family_rank_manifest_token,
        "source_family_rank_note_token": args.source_family_rank_note_token,
        "source_family_rank_extra_weight": args.source_family_rank_extra_weight,
        "source_family_rank_extra_manifest_token": args.source_family_rank_extra_manifest_token,
        "source_family_rank_extra_note_token": args.source_family_rank_extra_note_token,
        "source_family_real_guard_weight": args.source_family_real_guard_weight,
        "source_family_real_guard_min_sim": args.source_family_real_guard_min_sim,
        "source_family_real_guard_topk_frac": args.source_family_real_guard_topk_frac,
        "source_family_real_guard_manifest_token": args.source_family_real_guard_manifest_token,
        "source_family_real_guard_note_token": args.source_family_real_guard_note_token,
        "pair_group_prefix": args.pair_group_prefix,
        "include_protected_real_train": args.include_protected_real_train,
        "protected_real_aux_only": args.protected_real_aux_only,
        "pair_ignore_protected_real": args.pair_ignore_protected_real,
        "head_type": args.head_type,
        "cosine_scale_init": args.cosine_scale_init,
        "real_concept_prototypes": args.real_concept_prototypes,
        "real_concept_scale_init": args.real_concept_scale_init,
        "real_concept_weight": args.real_concept_weight,
        "real_concept_real_min_sim": args.real_concept_real_min_sim,
        "real_concept_fake_max_sim": args.real_concept_fake_max_sim,
        "real_concept_fake_weight": args.real_concept_fake_weight,
        "real_concept_use_sample_weights": args.real_concept_use_sample_weights,
        "targeted_fake_margin_weight": args.targeted_fake_margin_weight,
        "targeted_fake_margin_logit": args.targeted_fake_margin_logit,
        "targeted_fake_margin_manifest_token": args.targeted_fake_margin_manifest_token,
        "targeted_fake_margin_note_token": args.targeted_fake_margin_note_token,
        "targeted_real_margin_weight": args.targeted_real_margin_weight,
        "targeted_real_margin_logit": args.targeted_real_margin_logit,
        "targeted_real_margin_manifest_token": args.targeted_real_margin_manifest_token,
        "targeted_real_margin_note_token": args.targeted_real_margin_note_token,
        "targeted_real_margin_note_token_weight": args.targeted_real_margin_note_token_weight,
        "clip_mean": CLIP_MEAN,
        "clip_std": CLIP_STD,
        "replaced_linears": model.replaced_linears,
    }


class HighFrequencyEmphasis(nn.Module):
    def __init__(self, alpha: float, kernel_size: int = 5) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.kernel_size = int(kernel_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.alpha <= 0:
            return image
        squeeze = image.dim() == 3
        values = image.unsqueeze(0) if squeeze else image
        pad = self.kernel_size // 2
        padded = F.pad(values, (pad, pad, pad, pad), mode="reflect")
        blurred = F.avg_pool2d(padded, kernel_size=self.kernel_size, stride=1)
        enhanced = (values + self.alpha * (values - blurred)).clamp(0.0, 1.0)
        return enhanced.squeeze(0) if squeeze else enhanced


def weighted_loss_mean(losses: torch.Tensor, sample_weights: torch.Tensor | None = None) -> torch.Tensor:
    if sample_weights is None:
        return losses.mean()
    weights = sample_weights.to(device=losses.device, dtype=losses.dtype)
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return losses.sum() * 0.0
    return (losses * weights).sum() / weight_sum


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        losses = F.cross_entropy(logits, labels, weight=self.class_weights, reduction="none")
        return weighted_loss_mean(losses, sample_weights)


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma: float, class_weights: torch.Tensor) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, weight=self.class_weights, reduction="none")
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, labels.view(-1, 1)).squeeze(1).clamp(min=1e-6, max=1.0)
        return weighted_loss_mean(((1.0 - pt) ** self.gamma) * ce_loss, sample_weights)


class MarginAwareLoss(nn.Module):
    def __init__(
        self,
        base_loss: nn.Module,
        fake_margin_weight: float,
        fake_margin_logit: float,
        real_margin_weight: float,
        real_margin_logit: float,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.fake_margin_weight = fake_margin_weight
        self.fake_margin_logit = fake_margin_logit
        self.real_margin_weight = real_margin_weight
        self.real_margin_logit = real_margin_logit

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.base_loss(logits, labels, sample_weights=sample_weights)
        logit_margin = logits[:, 1] - logits[:, 0]
        fake_mask = labels == 1
        real_mask = labels == 0
        if self.fake_margin_weight > 0 and fake_mask.any():
            fake_weights = sample_weights[fake_mask] if sample_weights is not None else None
            fake_penalty = weighted_loss_mean(F.relu(self.fake_margin_logit - logit_margin[fake_mask]), fake_weights)
            loss = loss + self.fake_margin_weight * fake_penalty
        if self.real_margin_weight > 0 and real_mask.any():
            real_weights = sample_weights[real_mask] if sample_weights is not None else None
            real_penalty = weighted_loss_mean(F.relu(logit_margin[real_mask] + self.real_margin_logit), real_weights)
            loss = loss + self.real_margin_weight * real_penalty
        return loss


def make_criterion(args: argparse.Namespace) -> nn.Module:
    class_weights = torch.tensor(
        [args.real_class_weight, args.fake_class_weight],
        dtype=torch.float32,
        device=args.device,
    )
    if args.loss_type == "focal":
        base_loss: nn.Module = FocalCrossEntropyLoss(args.focal_gamma, class_weights)
    else:
        base_loss = WeightedCrossEntropyLoss(class_weights)
    if args.fake_margin_weight > 0 or args.real_margin_weight > 0:
        return MarginAwareLoss(
            base_loss=base_loss,
            fake_margin_weight=args.fake_margin_weight,
            fake_margin_logit=args.fake_margin_logit,
            real_margin_weight=args.real_margin_weight,
            real_margin_logit=args.real_margin_logit,
        )
    return base_loss


def patch_mil_loss(
    patch_logits: torch.Tensor,
    labels: torch.Tensor,
    topk_frac: float,
    real_weight: float,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    num_patches = patch_logits.shape[1]
    topk = max(1, min(num_patches, int(math.ceil(num_patches * topk_frac))))
    losses = []
    weights = []
    fake_mask = labels == 1
    real_mask = labels == 0
    if fake_mask.any():
        fake_logits = torch.topk(patch_logits[fake_mask], k=topk, dim=1).values
        fake_losses = F.binary_cross_entropy_with_logits(
            fake_logits,
            torch.ones_like(fake_logits),
            reduction="none",
        ).mean(dim=1)
        losses.append(fake_losses)
        if sample_weights is not None:
            weights.append(sample_weights[fake_mask])
    if real_mask.any():
        hard_real_logits = torch.topk(patch_logits[real_mask], k=topk, dim=1).values
        real_losses = F.binary_cross_entropy_with_logits(
            hard_real_logits,
            torch.zeros_like(hard_real_logits),
            reduction="none",
        ).mean(dim=1)
        losses.append(real_weight * real_losses)
        if sample_weights is not None:
            weights.append(sample_weights[real_mask])
    if not losses:
        return patch_logits.new_tensor(0.0)
    all_losses = torch.cat(losses, dim=0)
    all_weights = torch.cat(weights, dim=0) if weights else None
    return weighted_loss_mean(all_losses, all_weights)


def protected_real_patch_guard_loss(
    patch_logits: torch.Tensor | None,
    protected_real_flags: torch.Tensor | None,
    topk_frac: float,
) -> tuple[torch.Tensor, int, int]:
    if patch_logits is None or protected_real_flags is None:
        device = patch_logits.device if patch_logits is not None else "cpu"
        return torch.tensor(0.0, device=device), 0, 0
    protected_mask = protected_real_flags > 0
    candidate_count = int(protected_mask.sum().item())
    if candidate_count == 0:
        return patch_logits.new_tensor(0.0), 0, 0
    num_patches = patch_logits.shape[1]
    topk = max(1, min(num_patches, int(math.ceil(num_patches * topk_frac))))
    protected_logits = torch.topk(patch_logits[protected_mask], k=topk, dim=1).values
    losses = F.binary_cross_entropy_with_logits(
        protected_logits,
        torch.zeros_like(protected_logits),
        reduction="none",
    ).mean(dim=1)
    active_count = int((protected_logits.mean(dim=1) > 0).sum().item())
    return losses.mean(), active_count, candidate_count


def pair_consistency_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor | None,
    margin: float,
    fake_probabilities: torch.Tensor | None = None,
    hard_fake_prob_max: float = 1.0,
) -> tuple[torch.Tensor, int, int]:
    if group_ids is None:
        return features.new_tensor(0.0), 0, 0
    valid_groups = group_ids >= 0
    if not valid_groups.any():
        return features.new_tensor(0.0), 0, 0
    normalized = F.normalize(features, dim=-1)
    losses = []
    active_fake_count = 0
    candidate_fake_count = 0
    for group_id in group_ids[valid_groups].unique():
        group_mask = group_ids == group_id
        real_features = normalized[group_mask & (labels == 0)]
        fake_mask = group_mask & (labels == 1)
        if real_features.numel() == 0 or not fake_mask.any():
            continue
        candidate_fake_count += int(fake_mask.sum().item())
        if fake_probabilities is not None and hard_fake_prob_max < 1.0:
            fake_mask = fake_mask & (fake_probabilities <= hard_fake_prob_max)
        fake_features = normalized[fake_mask]
        if fake_features.numel() == 0:
            continue
        closest_source_real = fake_features @ real_features.detach().t()
        hard_similarity = closest_source_real.max(dim=1).values
        violations = F.relu(hard_similarity - margin)
        active_fake_count += int((violations > 0).sum().item())
        losses.append(violations.mean())
    if not losses:
        return features.new_tensor(0.0), active_fake_count, candidate_fake_count
    return torch.stack(losses).mean(), active_fake_count, candidate_fake_count


def source_family_ranking_loss(
    patch_tokens: torch.Tensor | None,
    labels: torch.Tensor,
    group_ids: torch.Tensor | None,
    margin: float,
    topk_frac: float,
    fake_probabilities: torch.Tensor | None = None,
    hard_fake_prob_max: float = 1.0,
    fake_rank_flags: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int]:
    if patch_tokens is None or group_ids is None:
        device = labels.device if patch_tokens is None else patch_tokens.device
        return torch.tensor(0.0, device=device), 0, 0
    valid_groups = group_ids >= 0
    if not valid_groups.any():
        return patch_tokens.new_tensor(0.0), 0, 0
    normalized = F.normalize(patch_tokens, dim=-1)
    num_patches = normalized.shape[1]
    topk = max(1, min(num_patches, int(math.ceil(num_patches * topk_frac))))
    losses = []
    active_fake_count = 0
    candidate_fake_count = 0
    for group_id in group_ids[valid_groups].unique():
        group_mask = group_ids == group_id
        real_tokens = normalized[group_mask & (labels == 0)]
        fake_mask = group_mask & (labels == 1)
        if fake_rank_flags is not None:
            fake_mask = fake_mask & (fake_rank_flags > 0)
        if real_tokens.numel() == 0 or not fake_mask.any():
            continue
        candidate_fake_count += int(fake_mask.sum().item())
        if fake_probabilities is not None and hard_fake_prob_max < 1.0:
            fake_mask = fake_mask & (fake_probabilities <= hard_fake_prob_max)
        fake_tokens = normalized[fake_mask]
        if fake_tokens.numel() == 0:
            continue
        source_anchor = F.normalize(real_tokens.detach().mean(dim=0), dim=-1)
        real_patch_scores = (real_tokens.detach() * source_anchor.unsqueeze(0)).sum(dim=-1)
        fake_patch_scores = (fake_tokens * source_anchor.unsqueeze(0)).sum(dim=-1)
        real_reference = torch.topk(real_patch_scores, k=topk, dim=1).values.mean(dim=1).mean().detach()
        fake_source_similarity = torch.topk(fake_patch_scores, k=topk, dim=1).values.mean(dim=1)
        violations = F.relu(fake_source_similarity - real_reference + margin)
        active_fake_count += int((violations > 0).sum().item())
        losses.append(violations.mean())
    if not losses:
        return patch_tokens.new_tensor(0.0), active_fake_count, candidate_fake_count
    return torch.stack(losses).mean(), active_fake_count, candidate_fake_count


def source_family_real_guard_loss(
    patch_tokens: torch.Tensor | None,
    labels: torch.Tensor,
    group_ids: torch.Tensor | None,
    min_similarity: float,
    topk_frac: float,
    real_guard_flags: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int]:
    if patch_tokens is None or group_ids is None:
        device = labels.device if patch_tokens is None else patch_tokens.device
        return torch.tensor(0.0, device=device), 0, 0
    valid_groups = group_ids >= 0
    if not valid_groups.any():
        return patch_tokens.new_tensor(0.0), 0, 0
    normalized = F.normalize(patch_tokens, dim=-1)
    num_patches = normalized.shape[1]
    topk = max(1, min(num_patches, int(math.ceil(num_patches * topk_frac))))
    losses = []
    active_real_count = 0
    candidate_real_count = 0
    for group_id in group_ids[valid_groups].unique():
        group_mask = group_ids == group_id
        all_real_tokens = normalized[group_mask & (labels == 0)]
        if all_real_tokens.numel() == 0:
            continue
        candidate_mask = group_mask & (labels == 0)
        if real_guard_flags is not None:
            candidate_mask = candidate_mask & (real_guard_flags > 0)
        real_tokens = normalized[candidate_mask]
        if real_tokens.numel() == 0:
            continue
        candidate_real_count += int(real_tokens.shape[0])
        source_anchor = F.normalize(all_real_tokens.detach().mean(dim=0), dim=-1)
        real_patch_scores = (real_tokens * source_anchor.unsqueeze(0)).sum(dim=-1)
        real_source_similarity = torch.topk(real_patch_scores, k=topk, dim=1).values.mean(dim=1)
        violations = F.relu(min_similarity - real_source_similarity)
        active_real_count += int((violations > 0).sum().item())
        losses.append(violations.mean())
    if not losses:
        return patch_tokens.new_tensor(0.0), active_real_count, candidate_real_count
    return torch.stack(losses).mean(), active_real_count, candidate_real_count


def real_concept_loss(
    model: EffortStyleDetector,
    features: torch.Tensor,
    labels: torch.Tensor,
    real_min_similarity: float,
    fake_max_similarity: float,
    fake_weight: float,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int, int, int]:
    if not isinstance(model.head, IndependentDualClassifier):
        return features.new_tensor(0.0), 0, 0, 0, 0
    normalized = model.head.normalized_features(features, apply_dropout=False)
    real_scores = normalized @ model.head.normalized_real_prototypes().t()
    nearest_real_similarity = real_scores.max(dim=1).values
    real_mask = labels == 0
    fake_mask = labels == 1
    losses = []
    weights = []
    active_reals = 0
    candidate_reals = int(real_mask.sum().item())
    active_fakes = 0
    candidate_fakes = int(fake_mask.sum().item())
    if real_mask.any():
        real_violations = F.relu(float(real_min_similarity) - nearest_real_similarity[real_mask])
        active_reals = int((real_violations > 0).sum().item())
        losses.append(real_violations)
        if sample_weights is not None:
            weights.append(sample_weights[real_mask])
    if fake_mask.any() and fake_weight > 0:
        fake_violations = F.relu(nearest_real_similarity[fake_mask] - float(fake_max_similarity))
        active_fakes = int((fake_violations > 0).sum().item())
        losses.append(float(fake_weight) * fake_violations)
        if sample_weights is not None:
            weights.append(sample_weights[fake_mask])
    if not losses:
        return features.new_tensor(0.0), active_reals, candidate_reals, active_fakes, candidate_fakes
    all_losses = torch.cat(losses, dim=0)
    all_weights = torch.cat(weights, dim=0) if weights else None
    return weighted_loss_mean(all_losses, all_weights), active_reals, candidate_reals, active_fakes, candidate_fakes


def targeted_fake_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    fake_margin_flags: torch.Tensor | None,
    margin_logit: float,
) -> tuple[torch.Tensor, int, int]:
    if fake_margin_flags is None:
        return logits.new_tensor(0.0), 0, 0
    candidate_mask = (labels == 1) & (fake_margin_flags > 0)
    candidate_count = int(candidate_mask.sum().item())
    if candidate_count == 0:
        return logits.new_tensor(0.0), 0, 0
    logit_margin = logits[:, 1] - logits[:, 0]
    violations = F.relu(float(margin_logit) - logit_margin[candidate_mask])
    active_count = int((violations > 0).sum().item())
    return violations.mean(), active_count, candidate_count


def targeted_real_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    real_margin_flags: torch.Tensor | None,
    margin_logit: float,
) -> tuple[torch.Tensor, int, int]:
    if real_margin_flags is None:
        return logits.new_tensor(0.0), 0, 0
    candidate_mask = (labels == 0) & (real_margin_flags > 0)
    candidate_count = int(candidate_mask.sum().item())
    if candidate_count == 0:
        return logits.new_tensor(0.0), 0, 0
    logit_margin = logits[:, 1] - logits[:, 0]
    violations = F.relu(logit_margin[candidate_mask] + float(margin_logit))
    active_count = int((violations > 0).sum().item())
    weights = real_margin_flags[candidate_mask].clamp_min(0.0)
    weighted_loss = (violations * weights).sum() / weights.sum().clamp_min(1e-6)
    return weighted_loss, active_count, candidate_count


def run_epoch(
    model: EffortStyleDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    orthogonal_weight: float,
    patch_mil_weight: float,
    patch_mil_topk_frac: float,
    patch_mil_real_weight: float,
    protected_real_patch_guard_weight: float,
    protected_real_patch_guard_topk_frac: float,
    pair_consistency_weight: float,
    pair_consistency_margin: float,
    pair_consistency_hard_fake_prob_max: float,
    source_family_rank_weight: float,
    source_family_rank_margin: float,
    source_family_rank_topk_frac: float,
    source_family_rank_hard_fake_prob_max: float,
    source_family_rank_extra_weight: float,
    source_family_real_guard_weight: float,
    source_family_real_guard_min_sim: float,
    source_family_real_guard_topk_frac: float,
    real_concept_weight: float,
    real_concept_real_min_sim: float,
    real_concept_fake_max_sim: float,
    real_concept_fake_weight: float,
    real_concept_use_sample_weights: bool,
    targeted_fake_margin_weight: float,
    targeted_fake_margin_logit: float,
    targeted_real_margin_weight: float,
    targeted_real_margin_logit: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_orthogonal = 0.0
    total_patch_mil = 0.0
    total_protected_real_patch_guard = 0.0
    total_pair = 0.0
    total_source_rank = 0.0
    total_source_real_guard = 0.0
    total_protected_real_patch_guard_active = 0
    total_protected_real_patch_guard_candidate = 0
    total_pair_active_fakes = 0
    total_pair_candidate_fakes = 0
    total_source_rank_active_fakes = 0
    total_source_rank_candidate_fakes = 0
    total_source_rank_extra = 0.0
    total_source_rank_extra_active_fakes = 0
    total_source_rank_extra_candidate_fakes = 0
    total_source_guard_active_reals = 0
    total_source_guard_candidate_reals = 0
    total_real_concept = 0.0
    total_real_concept_active_reals = 0
    total_real_concept_candidate_reals = 0
    total_real_concept_active_fakes = 0
    total_real_concept_candidate_fakes = 0
    total_targeted_fake_margin = 0.0
    total_targeted_fake_margin_active = 0
    total_targeted_fake_margin_candidate = 0
    total_targeted_real_margin = 0.0
    total_targeted_real_margin_active = 0
    total_targeted_real_margin_candidate = 0
    total_items = 0
    for batch in loader:
        classification_weights = None
        patch_mil_weights = None
        protected_real_flags = None
        source_rank_flags = None
        source_rank_extra_flags = None
        source_real_guard_flags = None
        targeted_fake_margin_flags = None
        targeted_real_margin_flags = None
        if len(batch) == 11:
            (
                images,
                labels,
                group_ids,
                classification_weights,
                patch_mil_weights,
                protected_real_flags,
                source_rank_flags,
                source_rank_extra_flags,
                source_real_guard_flags,
                targeted_fake_margin_flags,
                targeted_real_margin_flags,
            ) = batch
            group_ids = group_ids.to(device, non_blocking=True)
            classification_weights = classification_weights.to(device, non_blocking=True)
            patch_mil_weights = patch_mil_weights.to(device, non_blocking=True)
            protected_real_flags = protected_real_flags.to(device, non_blocking=True)
            source_rank_flags = source_rank_flags.to(device, non_blocking=True)
            source_rank_extra_flags = source_rank_extra_flags.to(device, non_blocking=True)
            source_real_guard_flags = source_real_guard_flags.to(device, non_blocking=True)
            targeted_fake_margin_flags = targeted_fake_margin_flags.to(device, non_blocking=True)
            targeted_real_margin_flags = targeted_real_margin_flags.to(device, non_blocking=True)
        elif len(batch) == 10:
            (
                images,
                labels,
                group_ids,
                classification_weights,
                patch_mil_weights,
                protected_real_flags,
                source_rank_flags,
                source_rank_extra_flags,
                source_real_guard_flags,
                targeted_fake_margin_flags,
            ) = batch
            group_ids = group_ids.to(device, non_blocking=True)
            classification_weights = classification_weights.to(device, non_blocking=True)
            patch_mil_weights = patch_mil_weights.to(device, non_blocking=True)
            protected_real_flags = protected_real_flags.to(device, non_blocking=True)
            source_rank_flags = source_rank_flags.to(device, non_blocking=True)
            source_rank_extra_flags = source_rank_extra_flags.to(device, non_blocking=True)
            source_real_guard_flags = source_real_guard_flags.to(device, non_blocking=True)
            targeted_fake_margin_flags = targeted_fake_margin_flags.to(device, non_blocking=True)
        elif len(batch) == 9:
            (
                images,
                labels,
                group_ids,
                classification_weights,
                patch_mil_weights,
                protected_real_flags,
                source_rank_flags,
                source_rank_extra_flags,
                source_real_guard_flags,
            ) = batch
            group_ids = group_ids.to(device, non_blocking=True)
            classification_weights = classification_weights.to(device, non_blocking=True)
            patch_mil_weights = patch_mil_weights.to(device, non_blocking=True)
            protected_real_flags = protected_real_flags.to(device, non_blocking=True)
            source_rank_flags = source_rank_flags.to(device, non_blocking=True)
            source_rank_extra_flags = source_rank_extra_flags.to(device, non_blocking=True)
            source_real_guard_flags = source_real_guard_flags.to(device, non_blocking=True)
        elif len(batch) == 8:
            (
                images,
                labels,
                group_ids,
                classification_weights,
                patch_mil_weights,
                protected_real_flags,
                source_rank_flags,
                source_rank_extra_flags,
            ) = batch
            group_ids = group_ids.to(device, non_blocking=True)
            classification_weights = classification_weights.to(device, non_blocking=True)
            patch_mil_weights = patch_mil_weights.to(device, non_blocking=True)
            protected_real_flags = protected_real_flags.to(device, non_blocking=True)
            source_rank_flags = source_rank_flags.to(device, non_blocking=True)
            source_rank_extra_flags = source_rank_extra_flags.to(device, non_blocking=True)
        elif len(batch) == 7:
            (
                images,
                labels,
                group_ids,
                classification_weights,
                patch_mil_weights,
                protected_real_flags,
                source_rank_flags,
            ) = batch
            group_ids = group_ids.to(device, non_blocking=True)
            classification_weights = classification_weights.to(device, non_blocking=True)
            patch_mil_weights = patch_mil_weights.to(device, non_blocking=True)
            protected_real_flags = protected_real_flags.to(device, non_blocking=True)
            source_rank_flags = source_rank_flags.to(device, non_blocking=True)
        elif len(batch) == 6:
            images, labels, group_ids, classification_weights, patch_mil_weights, protected_real_flags = batch
            group_ids = group_ids.to(device, non_blocking=True)
            classification_weights = classification_weights.to(device, non_blocking=True)
            patch_mil_weights = patch_mil_weights.to(device, non_blocking=True)
            protected_real_flags = protected_real_flags.to(device, non_blocking=True)
        elif len(batch) == 5:
            images, labels, group_ids, classification_weights, patch_mil_weights = batch
            group_ids = group_ids.to(device, non_blocking=True)
            classification_weights = classification_weights.to(device, non_blocking=True)
            patch_mil_weights = patch_mil_weights.to(device, non_blocking=True)
        elif len(batch) == 4:
            images, labels, group_ids, classification_weights = batch
            group_ids = group_ids.to(device, non_blocking=True)
            classification_weights = classification_weights.to(device, non_blocking=True)
        elif len(batch) == 3:
            images, labels, group_ids = batch
            group_ids = group_ids.to(device, non_blocking=True)
        else:
            images, labels = batch
            group_ids = None
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        use_aux = (
            patch_mil_weight > 0
            or protected_real_patch_guard_weight > 0
            or source_family_rank_weight > 0
            or source_family_rank_extra_weight > 0
            or source_family_real_guard_weight > 0
        )
        if use_aux:
            logits, features, aux = model(images, return_aux=True)
            if patch_mil_weight > 0:
                mil_loss = patch_mil_loss(
                    aux["patch_logits"],
                    labels,
                    topk_frac=patch_mil_topk_frac,
                    real_weight=patch_mil_real_weight,
                    sample_weights=patch_mil_weights,
                )
            else:
                mil_loss = next(model.parameters()).new_tensor(0.0)
            if protected_real_patch_guard_weight > 0:
                protected_real_guard_loss, protected_real_guard_active, protected_real_guard_candidate = (
                    protected_real_patch_guard_loss(
                        aux.get("patch_logits"),
                        protected_real_flags,
                        topk_frac=protected_real_patch_guard_topk_frac,
                    )
                )
            else:
                protected_real_guard_loss = next(model.parameters()).new_tensor(0.0)
                protected_real_guard_active = 0
                protected_real_guard_candidate = 0
        else:
            logits, features = model(images)
            aux = {}
            mil_loss = next(model.parameters()).new_tensor(0.0)
            protected_real_guard_loss = next(model.parameters()).new_tensor(0.0)
            protected_real_guard_active = 0
            protected_real_guard_candidate = 0
        fake_probabilities = torch.softmax(logits.detach(), dim=1)[:, 1]
        pair_loss, active_fakes, candidate_fakes = pair_consistency_loss(
            features,
            labels,
            group_ids,
            margin=pair_consistency_margin,
            fake_probabilities=fake_probabilities,
            hard_fake_prob_max=pair_consistency_hard_fake_prob_max,
        )
        source_rank_loss, source_rank_active_fakes, source_rank_candidate_fakes = source_family_ranking_loss(
            aux.get("patch_tokens"),
            labels,
            group_ids,
            margin=source_family_rank_margin,
            topk_frac=source_family_rank_topk_frac,
            fake_probabilities=fake_probabilities,
            hard_fake_prob_max=source_family_rank_hard_fake_prob_max,
            fake_rank_flags=source_rank_flags,
        )
        extra_source_rank_loss, extra_source_rank_active_fakes, extra_source_rank_candidate_fakes = (
            source_family_ranking_loss(
                aux.get("patch_tokens"),
                labels,
                group_ids,
                margin=source_family_rank_margin,
                topk_frac=source_family_rank_topk_frac,
                fake_probabilities=fake_probabilities,
                hard_fake_prob_max=source_family_rank_hard_fake_prob_max,
                fake_rank_flags=source_rank_extra_flags,
            )
        )
        real_guard_loss, real_guard_active_reals, real_guard_candidate_reals = source_family_real_guard_loss(
            aux.get("patch_tokens"),
            labels,
            group_ids,
            min_similarity=source_family_real_guard_min_sim,
            topk_frac=source_family_real_guard_topk_frac,
            real_guard_flags=source_real_guard_flags,
        )
        real_concept_aux_loss, real_concept_active_reals, real_concept_candidate_reals, real_concept_active_fakes, real_concept_candidate_fakes = real_concept_loss(
            model,
            features,
            labels,
            real_min_similarity=real_concept_real_min_sim,
            fake_max_similarity=real_concept_fake_max_sim,
            fake_weight=real_concept_fake_weight,
            sample_weights=classification_weights if real_concept_use_sample_weights else None,
        )
        targeted_fake_aux_loss, targeted_fake_active, targeted_fake_candidate = targeted_fake_margin_loss(
            logits,
            labels,
            targeted_fake_margin_flags,
            margin_logit=targeted_fake_margin_logit,
        )
        targeted_real_aux_loss, targeted_real_active, targeted_real_candidate = targeted_real_margin_loss(
            logits,
            labels,
            targeted_real_margin_flags,
            margin_logit=targeted_real_margin_logit,
        )
        ce_loss = criterion(logits, labels, sample_weights=classification_weights)
        orthogonal_loss = model.orthogonal_loss()
        loss = (
            ce_loss
            + orthogonal_weight * orthogonal_loss
            + patch_mil_weight * mil_loss
            + protected_real_patch_guard_weight * protected_real_guard_loss
            + pair_consistency_weight * pair_loss
            + source_family_rank_weight * source_rank_loss
            + source_family_rank_extra_weight * extra_source_rank_loss
            + source_family_real_guard_weight * real_guard_loss
            + real_concept_weight * real_concept_aux_loss
            + targeted_fake_margin_weight * targeted_fake_aux_loss
            + targeted_real_margin_weight * targeted_real_aux_loss
        )
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_ce += float(ce_loss.item()) * batch_size
        total_orthogonal += float(orthogonal_loss.item()) * batch_size
        total_patch_mil += float(mil_loss.item()) * batch_size
        total_protected_real_patch_guard += float(protected_real_guard_loss.item()) * batch_size
        total_pair += float(pair_loss.item()) * batch_size
        total_source_rank += float(source_rank_loss.item()) * batch_size
        total_source_rank_extra += float(extra_source_rank_loss.item()) * batch_size
        total_source_real_guard += float(real_guard_loss.item()) * batch_size
        total_real_concept += float(real_concept_aux_loss.item()) * batch_size
        total_targeted_fake_margin += float(targeted_fake_aux_loss.item()) * batch_size
        total_targeted_real_margin += float(targeted_real_aux_loss.item()) * batch_size
        total_protected_real_patch_guard_active += protected_real_guard_active
        total_protected_real_patch_guard_candidate += protected_real_guard_candidate
        total_pair_active_fakes += active_fakes
        total_pair_candidate_fakes += candidate_fakes
        total_source_rank_active_fakes += source_rank_active_fakes
        total_source_rank_candidate_fakes += source_rank_candidate_fakes
        total_source_rank_extra_active_fakes += extra_source_rank_active_fakes
        total_source_rank_extra_candidate_fakes += extra_source_rank_candidate_fakes
        total_source_guard_active_reals += real_guard_active_reals
        total_source_guard_candidate_reals += real_guard_candidate_reals
        total_real_concept_active_reals += real_concept_active_reals
        total_real_concept_candidate_reals += real_concept_candidate_reals
        total_real_concept_active_fakes += real_concept_active_fakes
        total_real_concept_candidate_fakes += real_concept_candidate_fakes
        total_targeted_fake_margin_active += targeted_fake_active
        total_targeted_fake_margin_candidate += targeted_fake_candidate
        total_targeted_real_margin_active += targeted_real_active
        total_targeted_real_margin_candidate += targeted_real_candidate
        total_items += batch_size
    return {
        "loss": total_loss / max(1, total_items),
        "ce_loss": total_ce / max(1, total_items),
        "orthogonal_loss": total_orthogonal / max(1, total_items),
        "patch_mil_loss": total_patch_mil / max(1, total_items),
        "protected_real_patch_guard_loss": total_protected_real_patch_guard / max(1, total_items),
        "protected_real_patch_guard_active_fraction": total_protected_real_patch_guard_active
        / max(1, total_protected_real_patch_guard_candidate),
        "protected_real_patch_guard_active_count": float(total_protected_real_patch_guard_active),
        "protected_real_patch_guard_candidate_count": float(total_protected_real_patch_guard_candidate),
        "pair_consistency_loss": total_pair / max(1, total_items),
        "pair_active_fake_fraction": total_pair_active_fakes / max(1, total_pair_candidate_fakes),
        "pair_active_fake_count": float(total_pair_active_fakes),
        "pair_candidate_fake_count": float(total_pair_candidate_fakes),
        "source_family_rank_loss": total_source_rank / max(1, total_items),
        "source_family_rank_active_fake_fraction": total_source_rank_active_fakes
        / max(1, total_source_rank_candidate_fakes),
        "source_family_rank_active_fake_count": float(total_source_rank_active_fakes),
        "source_family_rank_candidate_fake_count": float(total_source_rank_candidate_fakes),
        "source_family_rank_extra_loss": total_source_rank_extra / max(1, total_items),
        "source_family_rank_extra_active_fake_fraction": total_source_rank_extra_active_fakes
        / max(1, total_source_rank_extra_candidate_fakes),
        "source_family_rank_extra_active_fake_count": float(total_source_rank_extra_active_fakes),
        "source_family_rank_extra_candidate_fake_count": float(total_source_rank_extra_candidate_fakes),
        "source_family_real_guard_loss": total_source_real_guard / max(1, total_items),
        "source_family_real_guard_active_real_fraction": total_source_guard_active_reals
        / max(1, total_source_guard_candidate_reals),
        "source_family_real_guard_active_real_count": float(total_source_guard_active_reals),
        "source_family_real_guard_candidate_real_count": float(total_source_guard_candidate_reals),
        "real_concept_loss": total_real_concept / max(1, total_items),
        "real_concept_active_real_fraction": total_real_concept_active_reals
        / max(1, total_real_concept_candidate_reals),
        "real_concept_active_real_count": float(total_real_concept_active_reals),
        "real_concept_candidate_real_count": float(total_real_concept_candidate_reals),
        "real_concept_active_fake_fraction": total_real_concept_active_fakes
        / max(1, total_real_concept_candidate_fakes),
        "real_concept_active_fake_count": float(total_real_concept_active_fakes),
        "real_concept_candidate_fake_count": float(total_real_concept_candidate_fakes),
        "targeted_fake_margin_loss": total_targeted_fake_margin / max(1, total_items),
        "targeted_fake_margin_active_fraction": total_targeted_fake_margin_active
        / max(1, total_targeted_fake_margin_candidate),
        "targeted_fake_margin_active_count": float(total_targeted_fake_margin_active),
        "targeted_fake_margin_candidate_count": float(total_targeted_fake_margin_candidate),
        "targeted_real_margin_loss": total_targeted_real_margin / max(1, total_items),
        "targeted_real_margin_active_fraction": total_targeted_real_margin_active
        / max(1, total_targeted_real_margin_candidate),
        "targeted_real_margin_active_count": float(total_targeted_real_margin_active),
        "targeted_real_margin_candidate_count": float(total_targeted_real_margin_candidate),
    }


def ensure_results_tsv(path: Path) -> None:
    header = "\t".join(
        [
            "timestamp",
            "run_name",
            "model_name",
            "residual_rank",
            "epochs",
            "best_epoch",
            "val_auc",
            "val_balanced_acc",
            "test_auc",
            "test_balanced_acc",
            "min_cross_balanced_acc",
            "mean_cross_balanced_acc",
            "max_cross_auc",
            "status",
            "description",
        ]
    )
    if not path.exists() or path.read_text(encoding="utf-8").splitlines()[0] != header:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = project_root()
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    run_name = args.run_name or f"effort_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_records = []
    for manifest_value in args.train_manifest:
        manifest_path = resolve_path(manifest_value, root)
        train_records.extend(
            load_manifest_records(
                manifest_path,
                manifest_path.stem,
                root,
                max_rows=args.max_train_rows,
                include_protected_real=args.include_protected_real_train,
            )
        )
    split_records = split_train_records(train_records, args.seed)
    eval_specs = [parse_eval_manifest(value) for value in args.eval_manifest]
    eval_records = {
        name: load_manifest_records(resolve_path(path, root), name, root, max_rows=args.max_eval_rows)
        for name, path in eval_specs
    }

    train_tf, eval_tf = make_transforms(args.image_size, args.frequency_emphasis)
    pair_group_index, pair_group_stats = build_pair_group_index(
        split_records["train"],
        args.pair_group_prefix,
        ignore_protected_real=args.pair_ignore_protected_real,
    )
    if (
        args.pair_consistency_weight > 0
        or args.source_family_rank_weight > 0
        or args.source_family_rank_extra_weight > 0
        or args.source_family_real_guard_weight > 0
    ):
        train_loader = make_pair_loader(
            split_records["train"],
            train_tf,
            args.batch_size,
            args.workers,
            pair_group_index,
            args.seed,
            protected_real_aux_only=args.protected_real_aux_only,
            patch_mil_manifest_tokens=args.patch_mil_manifest_token,
            protected_real_patch_guard_manifest_tokens=args.protected_real_patch_guard_manifest_token,
            protected_real_patch_guard_note_tokens=args.protected_real_patch_guard_note_token,
            source_family_rank_manifest_tokens=args.source_family_rank_manifest_token,
            source_family_rank_note_tokens=args.source_family_rank_note_token,
            source_family_rank_extra_manifest_tokens=args.source_family_rank_extra_manifest_token,
            source_family_rank_extra_note_tokens=args.source_family_rank_extra_note_token,
            source_family_real_guard_manifest_tokens=args.source_family_real_guard_manifest_token,
            source_family_real_guard_note_tokens=args.source_family_real_guard_note_token,
            targeted_fake_margin_manifest_tokens=args.targeted_fake_margin_manifest_token,
            targeted_fake_margin_note_tokens=args.targeted_fake_margin_note_token,
            targeted_real_margin_manifest_tokens=args.targeted_real_margin_manifest_token,
            targeted_real_margin_note_tokens=args.targeted_real_margin_note_token,
            targeted_real_margin_note_token_weights=args.targeted_real_margin_note_token_weight,
        )
    else:
        train_loader = make_train_loader(
            split_records["train"],
            train_tf,
            args.batch_size,
            args.workers,
            shuffle=True,
            protected_real_aux_only=args.protected_real_aux_only,
            patch_mil_manifest_tokens=args.patch_mil_manifest_token,
            protected_real_patch_guard_manifest_tokens=args.protected_real_patch_guard_manifest_token,
            protected_real_patch_guard_note_tokens=args.protected_real_patch_guard_note_token,
            source_family_rank_manifest_tokens=args.source_family_rank_manifest_token,
            source_family_rank_note_tokens=args.source_family_rank_note_token,
            source_family_rank_extra_manifest_tokens=args.source_family_rank_extra_manifest_token,
            source_family_rank_extra_note_tokens=args.source_family_rank_extra_note_token,
            source_family_real_guard_manifest_tokens=args.source_family_real_guard_manifest_token,
            source_family_real_guard_note_tokens=args.source_family_real_guard_note_token,
            targeted_fake_margin_manifest_tokens=args.targeted_fake_margin_manifest_token,
            targeted_fake_margin_note_tokens=args.targeted_fake_margin_note_token,
            targeted_real_margin_manifest_tokens=args.targeted_real_margin_manifest_token,
            targeted_real_margin_note_tokens=args.targeted_real_margin_note_token,
            targeted_real_margin_note_token_weights=args.targeted_real_margin_note_token_weight,
        )
    val_loader = make_loader(split_records["val"], eval_tf, args.batch_size, args.workers, shuffle=False)
    test_loader = make_loader(split_records["test"], eval_tf, args.batch_size, args.workers, shuffle=False)
    eval_loaders = {
        name: make_loader(records, eval_tf, args.batch_size, args.workers, shuffle=False)
        for name, records in eval_records.items()
    }

    model = EffortStyleDetector(
        args.model_name,
        args.cache_dir,
        args.residual_rank,
        args.dropout,
        pooling=args.pooling,
        attn_hidden=args.attn_hidden,
        mean_gate_init=args.mean_gate_init,
        use_patch_mil=args.patch_mil_weight > 0 or args.protected_real_patch_guard_weight > 0,
        head_type=args.head_type,
        cosine_scale_init=args.cosine_scale_init,
        pool_topk_frac=args.pool_topk_frac,
        real_concept_prototypes=args.real_concept_prototypes,
        real_concept_scale_init=args.real_concept_scale_init,
    ).to(args.device)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
    criterion = make_criterion(args)

    best_state = None
    best_epoch = -1
    best_val_auc = -1.0
    history = []
    start = time.time()
    epoch_dir = run_dir / "epochs"
    if args.save_epoch_checkpoints:
        epoch_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_summary = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            args.device,
            args.orthogonal_weight,
            args.patch_mil_weight,
            args.patch_mil_topk_frac,
            args.patch_mil_real_weight,
            args.protected_real_patch_guard_weight,
            args.protected_real_patch_guard_topk_frac,
            args.pair_consistency_weight,
            args.pair_consistency_margin,
            args.pair_consistency_hard_fake_prob_max,
            args.source_family_rank_weight,
            args.source_family_rank_margin,
            args.source_family_rank_topk_frac,
            args.source_family_rank_hard_fake_prob_max,
            args.source_family_rank_extra_weight,
            args.source_family_real_guard_weight,
            args.source_family_real_guard_min_sim,
            args.source_family_real_guard_topk_frac,
            args.real_concept_weight,
            args.real_concept_real_min_sim,
            args.real_concept_fake_max_sim,
            args.real_concept_fake_weight,
            args.real_concept_use_sample_weights,
            args.targeted_fake_margin_weight,
            args.targeted_fake_margin_logit,
            args.targeted_real_margin_weight,
            args.targeted_real_margin_logit,
        )
        val_half = evaluate(model, val_loader, args.device, threshold=0.5)
        if val_half["auc"] > best_val_auc:
            best_val_auc = val_half["auc"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if args.save_epoch_checkpoints:
            epoch_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(checkpoint_payload(model, epoch_state, args), epoch_dir / f"epoch_{epoch:03d}.pt")
        history.append({"epoch": epoch, "train": train_summary, "val_at_0_5": val_half})
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    val_labels, val_probs = collect_probabilities(model, val_loader, args.device)
    threshold, threshold_balanced_acc = select_threshold(val_labels, val_probs)
    cross = {name: evaluate(model, loader, args.device, threshold) for name, loader in eval_loaders.items()}
    cross_ba = [value["balanced_acc"] for value in cross.values() if not math.isnan(value["balanced_acc"])]
    cross_auc = [value["auc"] for value in cross.values() if not math.isnan(value["auc"])]
    metrics = {
        "run_name": run_name,
        "model_name": args.model_name,
        "residual_rank": args.residual_rank,
        "orthogonal_weight": args.orthogonal_weight,
        "pooling": args.pooling,
        "attn_hidden": args.attn_hidden,
        "mean_gate_init": args.mean_gate_init,
        "pool_topk_frac": args.pool_topk_frac,
        "frequency_emphasis": args.frequency_emphasis,
        "loss_type": args.loss_type,
        "focal_gamma": args.focal_gamma,
        "real_class_weight": args.real_class_weight,
        "fake_class_weight": args.fake_class_weight,
        "fake_margin_weight": args.fake_margin_weight,
        "fake_margin_logit": args.fake_margin_logit,
        "real_margin_weight": args.real_margin_weight,
        "real_margin_logit": args.real_margin_logit,
        "patch_mil_weight": args.patch_mil_weight,
        "patch_mil_real_weight": args.patch_mil_real_weight,
        "patch_mil_topk_frac": args.patch_mil_topk_frac,
        "patch_mil_manifest_token": args.patch_mil_manifest_token,
        "protected_real_patch_guard_weight": args.protected_real_patch_guard_weight,
        "protected_real_patch_guard_topk_frac": args.protected_real_patch_guard_topk_frac,
        "protected_real_patch_guard_manifest_token": args.protected_real_patch_guard_manifest_token,
        "protected_real_patch_guard_note_token": args.protected_real_patch_guard_note_token,
        "use_patch_mil": args.patch_mil_weight > 0 or args.protected_real_patch_guard_weight > 0,
        "pair_consistency_weight": args.pair_consistency_weight,
        "pair_consistency_margin": args.pair_consistency_margin,
        "pair_consistency_hard_fake_prob_max": args.pair_consistency_hard_fake_prob_max,
        "source_family_rank_weight": args.source_family_rank_weight,
        "source_family_rank_margin": args.source_family_rank_margin,
        "source_family_rank_topk_frac": args.source_family_rank_topk_frac,
        "source_family_rank_hard_fake_prob_max": args.source_family_rank_hard_fake_prob_max,
        "source_family_rank_manifest_token": args.source_family_rank_manifest_token,
        "source_family_rank_note_token": args.source_family_rank_note_token,
        "source_family_rank_extra_weight": args.source_family_rank_extra_weight,
        "source_family_rank_extra_manifest_token": args.source_family_rank_extra_manifest_token,
        "source_family_rank_extra_note_token": args.source_family_rank_extra_note_token,
        "source_family_real_guard_weight": args.source_family_real_guard_weight,
        "source_family_real_guard_min_sim": args.source_family_real_guard_min_sim,
        "source_family_real_guard_topk_frac": args.source_family_real_guard_topk_frac,
        "source_family_real_guard_manifest_token": args.source_family_real_guard_manifest_token,
        "source_family_real_guard_note_token": args.source_family_real_guard_note_token,
        "pair_group_prefix": args.pair_group_prefix,
        "pair_group_stats": pair_group_stats,
        "include_protected_real_train": args.include_protected_real_train,
        "protected_real_aux_only": args.protected_real_aux_only,
        "pair_ignore_protected_real": args.pair_ignore_protected_real,
        "save_epoch_checkpoints": args.save_epoch_checkpoints,
        "head_type": args.head_type,
        "cosine_scale_init": args.cosine_scale_init,
        "real_concept_prototypes": args.real_concept_prototypes,
        "real_concept_scale_init": args.real_concept_scale_init,
        "real_concept_weight": args.real_concept_weight,
        "real_concept_real_min_sim": args.real_concept_real_min_sim,
        "real_concept_fake_max_sim": args.real_concept_fake_max_sim,
        "real_concept_fake_weight": args.real_concept_fake_weight,
        "real_concept_use_sample_weights": args.real_concept_use_sample_weights,
        "targeted_fake_margin_weight": args.targeted_fake_margin_weight,
        "targeted_fake_margin_logit": args.targeted_fake_margin_logit,
        "targeted_fake_margin_manifest_token": args.targeted_fake_margin_manifest_token,
        "targeted_fake_margin_note_token": args.targeted_fake_margin_note_token,
        "targeted_real_margin_weight": args.targeted_real_margin_weight,
        "targeted_real_margin_logit": args.targeted_real_margin_logit,
        "targeted_real_margin_manifest_token": args.targeted_real_margin_manifest_token,
        "targeted_real_margin_note_token": args.targeted_real_margin_note_token,
        "targeted_real_margin_note_token_weight": args.targeted_real_margin_note_token_weight,
        "replaced_linears": model.replaced_linears,
        "description": args.description,
        "device": args.device,
        "best_epoch": best_epoch,
        "selected_threshold": threshold,
        "selected_threshold_val_balanced_acc": threshold_balanced_acc,
        "counts": {
            "train": len(split_records["train"]),
            "val": len(split_records["val"]),
            "test": len(split_records["test"]),
            **{f"eval_{name}": len(records) for name, records in eval_records.items()},
        },
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable_parameters(model))),
        "val": evaluate(model, val_loader, args.device, threshold),
        "test": evaluate(model, test_loader, args.device, threshold),
        "cross": cross,
        "cross_summary": {
            "min_balanced_acc": min(cross_ba) if cross_ba else float("nan"),
            "mean_balanced_acc": float(np.mean(cross_ba)) if cross_ba else float("nan"),
            "max_auc": max(cross_auc) if cross_auc else float("nan"),
        },
        "elapsed_sec": round(time.time() - start, 2),
        "history": history,
    }

    checkpoint_path = run_dir / "best.pt"
    if best_state is not None:
        torch.save(checkpoint_payload(model, best_state, args), checkpoint_path)
    metrics["checkpoint"] = checkpoint_path.as_posix()
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    ensure_results_tsv(Path(args.results_tsv))
    append_results(
        Path(args.results_tsv),
        [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run_name,
            args.model_name,
            str(args.residual_rank),
            str(args.epochs),
            str(best_epoch),
            f"{metrics['val']['auc']:.6f}",
            f"{metrics['val']['balanced_acc']:.6f}",
            f"{metrics['test']['auc']:.6f}",
            f"{metrics['test']['balanced_acc']:.6f}",
            f"{metrics['cross_summary']['min_balanced_acc']:.6f}",
            f"{metrics['cross_summary']['mean_balanced_acc']:.6f}",
            f"{metrics['cross_summary']['max_auc']:.6f}",
            "complete",
            args.description,
        ],
    )
    print(json.dumps({"metrics": str(metrics_path), **metrics}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
