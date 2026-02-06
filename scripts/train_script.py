"""
Structured training loop built on Ignite.

This module keeps the original training behavior (gradient accumulation,
checkpointing, W&B logging, early stopping, Optuna pruning, etc.) but
organizes the flow into clearly named steps:

    1. Build optimizer and LR scheduler
    2. Create Ignite engines (train/validation/test)
    3. Attach metrics and event handlers
    4. Run the trainer and report the final objective

Use the public ``train`` function as the entry point.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Tuple, Optional, Any
from timeit import default_timer as timer

import optuna
import torch
import torch.optim as optim
import wandb
from torch import nn

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Metric, RunningAverage, Average, Accuracy


from utils.accuracy import RankAccuracy, RankAccuracy_withMargin
from utils.losses import compute_loss
from utils.log import log

from ignite.exceptions import NotComputableError
from train_utils import print_run_plan, normalize_gaze_mode

        
class EarlyStopper:
    """Simple epoch-level early stopping helper."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0, mode: str = "max", start_epoch: int = 1):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.start_epoch = int(start_epoch)

        self.best = None
        self.best_epoch = None
        self.bad_epochs = 0

    def _is_improvement(self, current: float) -> bool:
        if self.best is None:
            return True

        if self.mode == "max":
            return current > (self.best + self.min_delta)
        return current < (self.best - self.min_delta)

    def update(self, epoch: int, current: float) -> Tuple[bool, bool]:
        """Return ``(should_stop, improved)`` after seeing ``current`` metric."""
        improved = False

        if epoch < self.start_epoch:
            return False, improved

        if self._is_improvement(current):
            self.best = float(current)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            improved = True
            return False, improved

        self.bad_epochs += 1
        should_stop = self.bad_epochs >= self.patience
        return should_stop, improved


class WeightedAverage(Metric):
    """Compute a weighted average over engine outputs."""

    def __init__(self, value_fn, weight_fn, device=None):
        super().__init__(output_transform=lambda x: x, device=device)
        self.value_fn = value_fn
        self.weight_fn = weight_fn

    def reset(self) -> None:
        self._sum = torch.tensor(0.0, device=self._device)
        self._weight = torch.tensor(0.0, device=self._device)

    def update(self, output) -> None:
        value = torch.as_tensor(self.value_fn(output), device=self._device, dtype=torch.float32)
        weight = torch.as_tensor(self.weight_fn(output), device=self._device, dtype=torch.float32)
        self._sum += value * weight
        self._weight += weight

    def compute(self):
        if self._weight.item() == 0:
            return 0.0
        return (self._sum / self._weight).item()

class BackboneFreezeController:
    """
    Freeze backbone parameters for the first `freeze_epochs` epochs (0-based),
    then unfreeze them. Optimizer param groups are never modified.
    """

    def __init__(self, freeze_epochs: int, finetune: bool, backbone_params):
        self.freeze_epochs = int(max(0, freeze_epochs))
        self.finetune = bool(finetune)
        self.backbone_params = backbone_params
        self._unfrozen = False

    def start_frozen(self) -> bool:
        return self.finetune and self.freeze_epochs > 0

    def apply_initial_freeze(self) -> None:
        if not self.start_frozen():
            return
        for _, p in self.backbone_params:
            p.requires_grad = False

    def maybe_unfreeze(self, epoch0: int) -> bool:
        if not self.finetune or self._unfrozen:
            return False
        if int(epoch0) < self.freeze_epochs:
            return False

        for _, p in self.backbone_params:
            p.requires_grad = True

        self._unfrozen = True
        return True



# --------------------------------------------------------------------------------------------------------------------
# Data preparation helpers
# --------------------------------------------------------------------------------------------------------------------

def _set_gaze_bp(model: torch.nn.Module, enabled: bool) -> None:
    """
    Enables/disables gradient retention for attention maps inside the transformer,
    supporting both plain modules and wrapper modules (DataParallel/DDP-like).
    """
    if hasattr(model, "set_gaze_backprop"):
        model.set_gaze_backprop(bool(enabled))
        return

    if hasattr(model, "module") and hasattr(model.module, "set_gaze_backprop"):
        model.module.set_gaze_backprop(bool(enabled))
        return

def _resolve_gaze_flags(args) -> tuple[str, bool, bool, bool]:
    gaze_cfg = getattr(args, "gaze_cfg", None)

    if gaze_cfg is not None:
        gaze_mode = str(getattr(gaze_cfg, "mode", "disable"))
        use_gaze_any = bool(getattr(gaze_cfg, "load_gaze", False))
        use_gaze_kl = bool(getattr(gaze_cfg, "use_kl_in_loss", False))
        use_gaze_inj = bool(getattr(gaze_cfg, "inject", False))
        return gaze_mode, use_gaze_any, use_gaze_kl, use_gaze_inj

    gaze_mode = normalize_gaze_mode(getattr(args, "gaze_mode", None))
    w_kl = float(getattr(args, "attn_w", 0.0) or 0.0)

    use_gaze_inj = gaze_mode in ("guide", "align+gaze")
    kl_requested = gaze_mode in ("diag", "guide", "align", "align+gaze")
    use_gaze_kl = bool(gaze_mode in ("align", "align+gaze") and (w_kl > 0.0))
    use_gaze_any = bool(gaze_mode != "disable") and bool(use_gaze_inj or kl_requested)

    return gaze_mode, use_gaze_any, use_gaze_kl, use_gaze_inj


def _prepare_batch(
    data: Dict[str, torch.Tensor],
    device: torch.device,
    args,
) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
    input_left = data["image_l"].to(device)
    input_right = data["image_r"].to(device)

    label_r = data["score_r"].to(device).float()
    label_c = data["score_c"].to(device).long()

    _gaze_mode, use_gaze_any, _use_gaze_kl, use_gaze_inj = _resolve_gaze_flags(args)

    labels: Dict[str, torch.Tensor] = {"label_r": label_r, "label_c": label_c}

    gaze_l = gaze_r = has_eye_mask = None
    if use_gaze_any:
        if ("gaze_l" not in data) or ("gaze_r" not in data):
            raise KeyError("Batch missing gaze_l/gaze_r while gaze is enabled.")
        if "has_eyetracker" not in data:
            raise KeyError("Batch missing has_eyetracker while gaze is enabled.")

        gaze_l = data["gaze_l"].to(device)
        gaze_r = data["gaze_r"].to(device)
        has_eye_mask = data["has_eyetracker"].to(device)

        labels["gaze_l"] = gaze_l
        labels["gaze_r"] = gaze_r
        labels["has_eye_mask"] = has_eye_mask

    if use_gaze_inj:
        if gaze_l is None or gaze_r is None or has_eye_mask is None:
            raise ValueError("Gaze injection enabled but gaze tensors are missing from batch.")
        inputs: Tuple[torch.Tensor, ...] = (input_left, input_right, gaze_l, gaze_r, has_eye_mask)
    else:
        inputs = (input_left, input_right)

    return inputs, labels


def _build_metrics_output(
    args,
    forward_dict: Dict[str, Dict[str, torch.Tensor]],
    labels: Dict[str, torch.Tensor],
    loss: torch.Tensor,
    parts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Per-batch metrics dictionary returned by an Ignite engine step.

    Keys per batch:
      - Always:
          loss
      - rcnn:
          rank_left, rank_right, label
      - sscnn:
          logits, label
      - rsscnn:
          rank_left, rank_right, logits, label_r, label_c,
          loss_kl, loss_kl_weighted, w_kl_eff, gaze_count
    """
    out: Dict[str, Any] = {"loss": float(loss.item())}

    if args.model == "rcnn":
        out.update(
            {
                "rank_left": forward_dict["left"]["output"],
                "rank_right": forward_dict["right"]["output"],
                "label": labels["label_r"],
            }
        )
        return out

    if args.model == "sscnn":
        out.update(
            {
                "logits": forward_dict["logits"]["output"],
                "label": labels["label_c"].long(),
            }
        )
        return out

    if args.model == "rsscnn":
        out.update(
            {
                "rank_left": forward_dict["left"]["output"],
                "rank_right": forward_dict["right"]["output"],
                "logits": forward_dict["logits"]["output"],
                "label_r": labels["label_r"],
                "label_c": labels["label_c"],
            }
        )

        if parts is None:
            out["loss_kl"] = 0.0
            out["loss_kl_weighted"] = 0.0
            out["w_kl_eff"] = 0.0
            out["gaze_count"] = 0
            out["gaze_mode"] = str(getattr(args, "gaze_mode", getattr(args, "gaze", "off")))
            out["use_gaze_kl"] = 0.0
            return out

        loss_kl = parts.get("loss_kl", 0.0)
        loss_kl_weighted = parts.get("loss_kl_weighted", 0.0)

        out["loss_kl"] = float(loss_kl.detach().item()) if torch.is_tensor(loss_kl) else float(loss_kl)
        out["loss_kl_weighted"] = (
            float(loss_kl_weighted.detach().item()) if torch.is_tensor(loss_kl_weighted) else float(loss_kl_weighted)
        )
        out["w_kl_eff"] = float(parts.get("w_kl_eff", 0.0))
        out["gaze_count"] = int(parts.get("gaze_count", 0))
        out["gaze_mode"] = str(parts.get("gaze_mode", getattr(args, "gaze_mode", getattr(args, "gaze", "off"))))
        out["use_gaze_kl"] = float(parts.get("use_gaze_kl", 0.0))
        return out

    raise ValueError(f"Unsupported model type: {args.model}")

# --------------------------------------------------------------------------------------------------------------------
# Optimizer / scheduler helpers
# --------------------------------------------------------------------------------------------------------------------

def _split_parameters(
    net: torch.nn.Module,
) -> Tuple[
    List[Tuple[str, torch.nn.Parameter]],
    List[Tuple[str, torch.nn.Parameter]],
    List[Tuple[str, torch.nn.Parameter]],
]:
    """
    Split trainable parameters into three groups:
      - head_params: everything that is not backbone and not gaze-guidance
      - gii_params:  gaze-guidance modules (gii_layers, gaze_embedder)
      - backbone_params: parameters under net.backbone (or net.module.backbone)

    Grouping is done by name prefix to keep DataParallel compatibility.
    """
    head_params: List[Tuple[str, torch.nn.Parameter]] = []
    gii_params: List[Tuple[str, torch.nn.Parameter]] = []
    backbone_params: List[Tuple[str, torch.nn.Parameter]] = []

    gii_prefixes = (
        "gii_layers.",
        "gaze_embedder.",
        "module.gii_layers.",
        "module.gaze_embedder.",
    )

    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue

        is_backbone = name.startswith("backbone.") or name.startswith("module.backbone.")
        is_gii = name.startswith(gii_prefixes)

        if is_backbone:
            backbone_params.append((name, param))
        elif is_gii:
            gii_params.append((name, param))
        else:
            head_params.append((name, param))

    return head_params, gii_params, backbone_params


def _separate_decay(params: List[Tuple[str, torch.nn.Parameter]]):
    """
    Split named parameters into:
      - decay: weights (typically matrices)
      - no_decay: biases, norms, and 1D scale parameters

    Rule:
      - bias -> no_decay
      - norm-like names -> no_decay
      - 1D tensors (LayerNorm weight, scale vectors) -> no_decay
    """
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []

    for name, param in params:
        if not param.requires_grad:
            continue

        name_l = name.lower()
        is_bias = name_l.endswith(".bias") or ("bias" in name_l)
        is_norm = ("norm" in name_l) or ("ln" in name_l) or ("bn" in name_l)
        is_1d = (param.ndim == 1)

        if is_bias or is_norm or is_1d or ("len_sig" in name_l):
            no_decay.append(param)
        else:
            decay.append(param)

    return decay, no_decay


def _set_requires_grad(named_params: List[Tuple[str, torch.nn.Parameter]], flag: bool) -> None:
    for _, p in named_params:
        p.requires_grad = bool(flag)


def build_optimizer(
    args,
    net: torch.nn.Module,
    is_transformer: bool,
    head_params: List[Tuple[str, torch.nn.Parameter]],
    gii_params: List[Tuple[str, torch.nn.Parameter]],
    backbone_params: List[Tuple[str, torch.nn.Parameter]],
) -> Tuple[torch.optim.Optimizer, Dict, Optional["BackboneFreezeController"]]:
    """
    Optimizer builder with:
      - head group (always)
      - optional GII group (always if present)
      - optional backbone group (if finetune=True)
      - optional backbone freeze for first N epochs (scheduler-stable)

    LRs:
      - head_lr = base_lr
      - gii_lr = base_lr * gii_lr_scale
      - backbone_lr = base_lr * backbone_lr_scale
    """
    base_lr = float(getattr(args, "base_lr"))
    weight_decay = float(getattr(args, "weight_decay", 0.0))
    finetune = bool(getattr(args, "finetune", False))

    bb_scale = float(getattr(args, "backbone_lr_scale", 0.1))
    gii_scale = float(getattr(args, "gii_lr_scale", 1.0))

    head_lr = base_lr
    gii_lr = base_lr * gii_scale
    backbone_lr = base_lr * bb_scale

    freeze_epochs = int(getattr(args, "backbone_freeze_epochs", 0))

    controller = BackboneFreezeController(
        freeze_epochs=freeze_epochs,
        finetune=finetune,
        backbone_params=backbone_params,
    )

    groups: List[Dict] = []

    # -------------------------
    # Head groups
    # -------------------------
    head_decay, head_no_decay = _separate_decay(head_params)
    if head_decay:
        groups.append({"params": head_decay, "lr": head_lr, "weight_decay": weight_decay})
    if head_no_decay:
        groups.append({"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0})

    # -------------------------
    # GII groups (guide modules)
    # -------------------------
    gii_decay, gii_no_decay = _separate_decay(gii_params)
    if gii_decay:
        groups.append({"params": gii_decay, "lr": gii_lr, "weight_decay": weight_decay})
    if gii_no_decay:
        groups.append({"params": gii_no_decay, "lr": gii_lr, "weight_decay": 0.0})

    # -------------------------
    # Backbone groups (only if finetune)
    # -------------------------
    if finetune:
        bb_decay, bb_no_decay = _separate_decay(backbone_params)
        if bb_decay:
            groups.append({"params": bb_decay, "lr": backbone_lr, "weight_decay": weight_decay})
        if bb_no_decay:
            groups.append({"params": bb_no_decay, "lr": backbone_lr, "weight_decay": 0.0})

    # -------------------------
    # Optimizer family
    # -------------------------
    if is_transformer:
        optimizer = optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        family, opt_name = "transformer", "AdamW"
    else:
        optimizer = optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8)
        family, opt_name = "cnn", "Adam"

    # Keeps optimizer param groups stable for schedulers
    controller.apply_initial_freeze()

    mode = (
        f"{family}_head_only"
        if not finetune
        else (f"{family}_freeze_backbone_{freeze_epochs}ep" if controller.start_frozen() else f"{family}_split_lr")
    )

    optimizer_info = {
        "family": family,
        "optimizer": opt_name,
        "mode": mode,
        "finetune": finetune,
        "head_lr": head_lr,
        "gii_lr": gii_lr,
        "backbone_lr": backbone_lr,
        "backbone_lr_scale": bb_scale,
        "gii_lr_scale": gii_scale,
        "weight_decay": weight_decay,
        "backbone_freeze_epochs": freeze_epochs,
        "n_head_params": len(head_params),
        "n_gii_params": len(gii_params),
        "n_backbone_params": len(backbone_params),
    }

    if not finetune:
        return optimizer, optimizer_info, None
    return optimizer, optimizer_info, controller

        
def _build_scheduler(args, optimizer, accum_steps: int, steps_per_epoch: int, base_lr: float):
    """
    Build the LR scheduler used by the Ignite training loop.

    Key concept: "optimizer steps" vs "dataloader iterations"
      - The program support gradient accumulation (args.k).
      - That means optimizer.step() happens only every accum_steps iterations.
      - Schedulers that step per-optimizer-step must be configured using the number of *optimizer steps*,
        not raw dataloader batches.

    Computation:
      - eff_steps_per_epoch = ceil(steps_per_epoch / accum_steps)
            steps_per_epoch is len(dataloader) (number of batches per epoch).
            dividing by accum_steps gives number of optimizer updates per epoch.
      - total_iters = args.max_epochs * eff_steps_per_epoch
            total number of optimizer updates over the entire run.

    Scheduler selection:
      - args.scheduler chooses the strategy.
      - Returns:
          (scheduler_instance_or_None, scheduler_type_string)
    """
    # Effective number of optimizer updates per epoch (accounts for gradient accumulation).
    eff_steps_per_epoch = math.ceil(steps_per_epoch / accum_steps)

    # Total number of optimizer steps across all epochs (this is the "time axis" for most schedulers).
    total_iters = args.max_epochs * eff_steps_per_epoch

    scheduler_type = args.scheduler
    scheduler = None

    # ------------------------------------------------------------------
    # 0) No scheduler: constant LR
    # ------------------------------------------------------------------
    if scheduler_type == "none":
        scheduler = None

    # ------------------------------------------------------------------
    # 1) Warmup + Cosine decay (implemented via LambdaLR)
    # ------------------------------------------------------------------
    elif scheduler_type == "warmup_cosine":
        # Warmup fraction is clamped to [0, 1] to avoid misconfiguration.
        warmup_frac = max(0.0, min(1.0, args.warmup_frac))
        warmup_iters = int(warmup_frac * total_iters)

        # eta_min is the absolute minimal LR (not a multiplier) used as the cosine floor.
        eta_min = args.eta_min

        def lr_lambda(step: int):
            """
            LambdaLR expects a multiplicative factor applied to each param_group's initial LR.

            Behavior:
              - During warmup: linear ramp from 0 → 1 over warmup_iters steps
              - After warmup: cosine decay from 1 → (eta_min/base_lr)
            """
            # Linear warmup phase: increase LR proportionally with step index.
            if warmup_iters > 0 and step < warmup_iters:
                return float(step) / float(max(1, warmup_iters))

            # Degenerate case: if warmup consumes all steps, hold at eta_min/base_lr.
            if total_iters == warmup_iters:
                return eta_min / base_lr

            # Cosine decay phase over the remaining steps.
            progress = min(1.0, max(0.0, (step - warmup_iters) / float(max(1, total_iters - warmup_iters))))

            # Convert eta_min to a multiplier relative to base_lr.
            eta_min_factor = eta_min / base_lr

            # Cosine interpolation between 1.0 and eta_min_factor.
            return eta_min_factor + (1 - eta_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ------------------------------------------------------------------
    # 2) Cosine decay (no warmup)
    # ------------------------------------------------------------------
    elif scheduler_type == "cosine":
        # Standard cosine annealing from initial LR to eta_min over total_iters steps.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_iters,
            eta_min=args.eta_min,
        )

    # ------------------------------------------------------------------
    # 3) OneCycleLR
    # ------------------------------------------------------------------
    elif scheduler_type == "onecycle":
        # OneCycleLR requires max_lr per param group.
        # It's set max_lr to the optimizer's current LR per group, so the schedule
        # oscillates around those group-specific settings.
        max_lrs = [pg["lr"] for pg in optimizer.param_groups]

        # pct_start controls fraction of the cycle spent increasing LR.
        warmup_frac = max(0.0, min(1.0, args.warmup_frac))
        pct_start = warmup_frac if warmup_frac > 0 else 0.3

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_iters,
            pct_start=pct_start,
            anneal_strategy="cos",
            cycle_momentum=False,  # Adam/AdamW style; momentum cycling not used
        )

    # ------------------------------------------------------------------
    # 4) CosineAnnealingWarmRestarts
    # ------------------------------------------------------------------
    elif scheduler_type == "warm_restarts":
        # Warm restarts: cosine cycles of length T_0, multiplied by T_mult each restart.
        # IMPORTANT: This scheduler uses "scheduler.step()" calls as its time steps.
        # In the training loop, it's called scheduler.step() every optimizer update
        # (except for plateau), so T_0 is interpreted in optimizer-step units.
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.T_0,
            T_mult=args.T_mult,
            eta_min=args.eta_min,
        )

    # ------------------------------------------------------------------
    # 5) ReduceLROnPlateau
    # ------------------------------------------------------------------
    elif scheduler_type == "plateau":
        # Plateau scheduler steps on a validation metric (here: mode="max" for accuracy).
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.plateau_min_lr,
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler, scheduler_type


# ------------------------------------------------------------------------------------------------------------------
# Engine step factories
# ------------------------------------------------------------------------------------------------------------------

def _make_train_step(
    args,
    device: torch.device,
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scheduler_type: str,
    accum_steps: int,
    grad_clip: float,
    logger,
    trial,
):
    """
    Ignite training-step factory.

    Centralized gaze behavior:
      - _prepare_batch uses args.gaze_cfg to decide whether gaze tensors are present and whether
        gaze is injected into the model forward.
      - compute_loss uses args.gaze_cfg to decide whether KL(attn↔gaze) is computed and whether
        it contributes to loss_total.

    Supported gaze modes:
      - disable     : no gaze tensors, no KL diagnostics, no injection
      - diag        : KL computed for diagnostics only
      - guide       : gaze injection enabled; KL computed for diagnostics only
      - align       : KL weighted into objective via attn_w
      - align+gaze  : injection enabled + KL weighted into objective via attn_w
    """
    accum = max(1, int(accum_steps))

    def train_step(engine, data):
        if os.path.exists("SKIP_TRIAL"):
            print("[USER REQUEST] SKIPPING THIS RUN NOW.")
            os.remove("SKIP_TRIAL")
            if trial is not None:
                raise optuna.TrialPruned()
            engine.terminate()
            return {"skipped": True}

        if logger:
            start = timer()

        inputs, labels = _prepare_batch(data, device, args)

        forward_dict = net(*inputs)
        loss_total, parts = compute_loss(args, forward_dict, labels, return_parts=True)

        if not torch.isfinite(loss_total.detach()).item():
            label_r = labels["label_r"]
            n_ties = (label_r == 0).sum().item()
            n_nonties = (label_r != 0).sum().item()
            print(
                f"[NaN/Inf DETECTED] epoch={engine.state.epoch} "
                f"iter={engine.state.iteration} "
                f"batch_size={label_r.size(0)}, "
                f"n_nonties={n_nonties}, n_ties={n_ties}"
            )
            raise ValueError("Non-finite loss detected; stopping for debugging.")

        loss_scaled = loss_total / accum

        loss_scaled.backward()

        if engine.state.iteration % accum == 0:
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)

            optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler and scheduler_type != "plateau":
                scheduler.step()

        if logger:
            logger.info(f"TRAIN_STEP, {timer() - start:.4f}")

        out = _build_metrics_output(
            args=args,
            forward_dict=forward_dict,
            labels=labels,
            loss=loss_total.detach(),
            parts=parts,
        )
        return out

    return train_step



def _make_inference_step(args, device: torch.device, net: torch.nn.Module):
    """
    Ignite inference-step factory for validation/test.

    Behavior:
      - Runs under torch.no_grad()
      - Calls compute_loss and reports loss_total (KL included only in align modes via w_kl_eff)
      - If gaze_mode includes guide, gaze tensors are passed into the transformer for injection
        (handled by _prepare_batch input packing)
    """
    def inference_step(engine, data):
        with torch.no_grad():
            inputs, labels = _prepare_batch(data, device, args)

            forward_dict = net(*inputs)
            loss_total, parts = compute_loss(args, forward_dict, labels, return_parts=True)

            out = _build_metrics_output(
                args=args,
                forward_dict=forward_dict,
                labels=labels,
                loss=loss_total,
                parts=parts,
            )
            return out

    return inference_step

# --------------------------------------------------------------------------------------------------------------------
# Metric + handler helpers
# --------------------------------------------------------------------------------------------------------------------

def _attach_metrics(engines: List[Engine], args, device: torch.device) -> None:
    """
    Attach Ignite metrics to one or more engines (typically: train / validation / test).

    When this is activated:
      - This function is called during pipeline setup, after the Ignite Engine(s) are created
        and before engine.run(...) is executed.
      - It is executed once per program run (or once per re-build of engines), not every iteration.
      - The attached metrics are then computed automatically by Ignite during engine execution:
          * RunningAverage updates every iteration (batch) for the current engine.
          * Accuracy / RankAccuracy accumulate across the epoch and are finalized at EPOCH_COMPLETED.

    Why it exists:
      - Different model modes (rcnn / sscnn / rsscnn) emit different outputs
        (ranking scores vs classification logits), so the correct metrics must be
        attached based on args.model.
      - "full_accuracy" toggles whether ranking accuracy is computed with an
        explicit margin criterion or as a pure ordering comparison.

    Expected engine output dictionary keys:
      - rcnn:   {"loss", "rank_left", "rank_right", "label", ...}
      - sscnn:  {"loss", "logits", "label", ...}
      - rsscnn: {"loss", "rank_left", "rank_right", "label_r", "logits", "label_c", ...}

    Notes on metric names:
      - "loss"  : RunningAverage over per-iteration loss values (smoothed training curve).
      - "acc"   : ranking accuracy (rcnn or rsscnn ranking branch).
      - "c_acc" : classification accuracy (rsscnn classification branch only).
    """
    for engine in engines:
        # ---------------------------------------------------------------------
        # RCNN: ranking-only training/evaluation (no classification head metric).
        # ---------------------------------------------------------------------
        if args.model == "rcnn":
            # RunningAverage tracks a smoothed loss over iterations for this engine.
            RunningAverage(output_transform=lambda x: x["loss"], device=device).attach(engine, "loss")

            # Ranking accuracy can optionally enforce a margin criterion.
            # - full_accuracy=True  : prediction is only correct if the score difference
            #                         exceeds args.ranking_margin in the correct direction.
            # - full_accuracy=False : prediction is correct if ordering matches the label,
            #                         ignoring margin magnitude.
            if args.full_accuracy:
                RankAccuracy_withMargin(
                    output_transform=lambda x: (x["rank_left"], x["rank_right"], x["label"], args.ranking_margin),
                    device=device,
                ).attach(engine, "acc")
            else:
                RankAccuracy(
                    output_transform=lambda x: (x["rank_left"], x["rank_right"], x["label"]),
                    device=device,
                ).attach(engine, "acc")

        # ---------------------------------------------------------------------
        # SSCNN: classification-only training/evaluation (no ranking metric).
        # ---------------------------------------------------------------------
        elif args.model == "sscnn":
            RunningAverage(output_transform=lambda x: x["loss"], device=device).attach(engine, "loss")

            # Standard multiclass accuracy using predicted logits vs ground-truth label.
            # Assumes logits shape [B, num_classes] and label shape [B].
            Accuracy(output_transform=lambda x: (x["logits"], x["label"])).attach(engine, "acc")

        # ---------------------------------------------------------------------
        # RSSCNN: ranking + classification + gaze
        # ---------------------------------------------------------------------
        elif args.model == "rsscnn":
            # 1) Loss (rolling average)
            RunningAverage(output_transform=lambda x: x["loss"], device=device).attach(engine, "loss")

            # 2) Gaze/KL diagnostics (always attach; KL is always computed in compute_loss)
            #    In guide/off modes, loss_kl_weighted stays 0 because w_kl_eff=0.
            WeightedAverage(
                value_fn=lambda x: x.get("loss_kl", 0.0),
                weight_fn=lambda x: x.get("gaze_count", 0),
                device=device,
            ).attach(engine, "loss_kl")
            WeightedAverage(
                value_fn=lambda x: x.get("loss_kl_weighted", 0.0),
                weight_fn=lambda x: x.get("gaze_count", 0),
                device=device,
            ).attach(engine, "loss_kl_weighted")
            RunningAverage(output_transform=lambda x: x.get("w_kl_eff", 0.0), device=device).attach(engine, "w_kl_eff")

            # Optional but strongly recommended: confirms whether any gaze samples exist in batches
            RunningAverage(output_transform=lambda x: float(x.get("gaze_count", 0)), device=device).attach(engine, "gaze_count")

            # Optional: track which mode is active in logs (string; keep in output dict but not as metric)
            # gaze_mode, *_ = _resolve_gaze_flags(args)

            # 3) Ranking accuracy
            if args.full_accuracy:
                RankAccuracy_withMargin(
                    output_transform=lambda x: (x["rank_left"], x["rank_right"], x["label_r"], args.ranking_margin),
                    device=device,
                ).attach(engine, "acc")
            else:
                RankAccuracy(
                    output_transform=lambda x: (x["rank_left"], x["rank_right"], x["label_r"]),
                    device=device,
                ).attach(engine, "acc")

            # 4) Classification accuracy
            Accuracy(output_transform=lambda x: (x["logits"], x["label_c"])).attach(engine, "c_acc")

        # ---------------------------------------------------------------------
        # Defensive programming: reject unknown model identifiers early.
        # ---------------------------------------------------------------------
        else:
            raise ValueError(f"Unsupported model type: {args.model}")

            
def _compute_class_breakdown(args, net, loader, device, split_name: str, epoch_idx: int, print_output: bool = True):
    """
    Computes a confusion matrix adapted to:
    - ties=True  → 3 classes  (0=left, 1=tie, 2=right)
    - ties=False → 2 classes  (0=left, 1=right)

    If print_output=False → do NOT print anything (useful for test set).
    """

    if args.model not in ["sscnn", "rsscnn"]:
        return None

    if args.ties:
        num_classes = 3
        class_names = ["left", "tie", "right"]
    else:
        num_classes = 2
        class_names = ["left", "right"]

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    # Resolve gaze flags once (mode does not change within a run)
    gaze_mode, use_gaze_any, use_gaze_kl, use_gaze_inj = _resolve_gaze_flags(args)

    net.eval()
    with torch.no_grad():
        for batch in loader:
            input_left = batch["image_l"].to(device)
            input_right = batch["image_r"].to(device)
            label_c = batch["score_c"].to(device).long()

            # ---------------------------------------------------------------------------------
            # Forward pass:
            #   - off / align      : image-only model execution
            #   - guide / align+guide : pass gaze + mask for injection
            # ---------------------------------------------------------------------------------
            if use_gaze_inj:
                if ("gaze_l" not in batch) or ("gaze_r" not in batch):
                    raise KeyError("Batch missing gaze tensors while gaze injection is enabled.")
                if "has_eyetracker" not in batch:
                    raise KeyError("Batch missing 'has_eyetracker' while gaze injection is enabled.")

                gaze_l = batch["gaze_l"].to(device)
                gaze_r = batch["gaze_r"].to(device)
                has_eye_mask = batch["has_eyetracker"].to(device)

                forward_dict = net(input_left, input_right, gaze_l, gaze_r, has_eye_mask)
            else:
                forward_dict = net(input_left, input_right)

            logits = forward_dict["logits"]["output"]
            preds = torch.argmax(logits, dim=1)

            for true_cls, pred_cls in zip(label_c.view(-1), preds.view(-1)):
                t = int(true_cls.item())
                p = int(pred_cls.item())
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    confusion[t, p] += 1

    if not print_output:
        return confusion

    print(f"\n[Epoch {epoch_idx}] {split_name} classification breakdown (true x pred)")
    print("Rows = true class, Cols = predicted class")
    print("Class mapping:", {i: name for i, name in enumerate(class_names)})
    print(confusion.cpu().numpy())

    for cls in range(num_classes):
        row = confusion[cls]
        total = int(row.sum().item())
        correct = int(row[cls].item())
        incorrect = total - correct

        if total == 0:
            print(f"  True class {cls} ({class_names[cls]}): no samples in this split.")
            continue

        print(
            f"\n  True class {cls} ({class_names[cls]}): total={total}, "
            f"correct={correct}, incorrect={incorrect} "
            f"({incorrect/total:.3f} misclass rate)"
        )

        if incorrect > 0:
            for pred_cls in range(num_classes):
                if pred_cls == cls:
                    continue
                count = int(row[pred_cls].item())
                if count > 0:
                    print(
                        f"    misclassified as {pred_cls} ({class_names[pred_cls]}): "
                        f"{count} ({count/incorrect:.3f} of misclassified)"
                    )

    print()
    return confusion


# --------------------------------------------------------------------------------------------------------------------
# Validation / logging handlers
# --------------------------------------------------------------------------------------------------------------------

def _attach_epoch_end_step(trainer: Engine, optimizer, accum_steps: int):
    """
    Flush a partially accumulated gradient at epoch end.

    With gradient accumulation, optimizer.step() runs only every accum_steps iterations.
    If the epoch ends mid-window, it's aplied one final step.

    """
    @trainer.on(Events.EPOCH_COMPLETED)
    def step_on_epoch_end(engine):
        if engine.state.iteration % accum_steps != 0:
            optimizer.step()

            optimizer.zero_grad(set_to_none=True)

def _make_validation_handler(
    args,
    net,
    trainer: Engine,
    evaluator: Engine,
    evaluator_test: Engine,
    optimizer,
    scheduler,
    scheduler_type: str,
    val_loader,
    test_loader,
    early_stopper: EarlyStopper | None,
    start_training: float,
    training_state: Dict[str, float | List[float]],
    device: torch.device,
):
    """
    Build and return an epoch-end validation/logging callback.

    When this is activated:
      - This function is called once during pipeline setup to create a handler function.
      - The returned 'log_validation_results' function is typically attached to the trainer as:
            trainer.on(Events.EPOCH_COMPLETED)(log_validation_results)
        (or equivalent).
      - As a result, 'log_validation_results' runs once per epoch at EPOCH_COMPLETED.

    Responsibilities of the handler:
      1) Run evaluation on validation and test dataloaders via separate evaluator engines.
      2) Update trainer state with validation metrics used by external components (e.g., Optuna pruning).
      3) Step schedulers that require validation feedback (ReduceLROnPlateau).
      4) Track best validation accuracy across epochs.
      5) Report intermediate results to Optuna and optionally prune trials.
      6) Optionally compute extra per-class / per-label breakdowns for classification models.
      7) Perform model mode switches (eval/train) and optional partial_eval hooks.
      8) Apply early stopping and terminate training if criteria are met.
      9) Emit a consolidated metrics dictionary through the project's logging utility.
    """
    
    def log_validation_results(engine):
        # ---------------------------------------------------------------------
        # 1) Switch model to evaluation mode and run evaluators
        # ---------------------------------------------------------------------
        net.eval()

        # Evaluator engines compute metrics over the full dataloader and store them
        # in evaluator.state.metrics (and evaluator_test.state.metrics).
        evaluator.run(val_loader)
        evaluator_test.run(test_loader)

        # Persist validation accuracy in trainer metrics for external consumers.
        engine.state.metrics["val_acc"] = evaluator.state.metrics["acc"]

        current_val_acc = float(evaluator.state.metrics["acc"])
        current_train_acc = engine.state.metrics.get("acc")
        current_test_acc = evaluator_test.state.metrics.get("acc")

        # ---------------------------------------------------------------------
        # 2) Scheduler stepping (only for validation-dependent schedulers)
        # ---------------------------------------------------------------------
        # ReduceLROnPlateau requires a monitored metric value; it should be stepped
        # once per epoch after validation, not every iteration.
        if scheduler and scheduler_type == "plateau":
            scheduler.step(current_val_acc)

            # Optional console sanity print: helps confirm plateau is receiving the metric
            # and that LR updates occur as expected.
            lr_head = optimizer.param_groups[0]["lr"]
            print(
                f"[Plateau sanity] "
                f"epoch={engine.state.epoch} "
                f"val_acc={current_val_acc:.4f} "
                f"lr={lr_head:.3e}"
            )

        # ---------------------------------------------------------------------
        # 3) Track validation accuracy history and best-so-far (selection-coupled)
        # ---------------------------------------------------------------------
        training_state["val_acc_history"].append(current_val_acc)
        
        if current_val_acc > float(training_state["best_val_acc"]):
            training_state["best_val_acc"] = float(current_val_acc)
            training_state["epoch_best_val"] = int(trainer.state.epoch)
        
            if current_train_acc is not None:
                training_state["train_acc_at_best_val"] = float(current_train_acc)
            if current_test_acc is not None:
                training_state["test_acc_at_best_val"] = float(current_test_acc)
        
            # Snapshot best weights for final evaluation (best epoch, not last epoch)
            #model_to_save = net.module if hasattr(net, "module") else net
            #training_state["best_state_dict"] = {
            #    k: v.detach().cpu().clone() for k, v in model_to_save.state_dict().items()
            #}
            training_state["best_state_dict"] = None

        # ---------------------------------------------------------------------
        # 4) Optuna integration (report + pruning)
        # ---------------------------------------------------------------------
        # If this run is being driven by an Optuna trial, report the current best
        # objective value and allow Optuna to prune unpromising runs.
        if engine.state.trial is not None:
            current_epoch = engine.state.epoch
            engine.state.trial.report(training_state["best_val_acc"], step=current_epoch)
            if engine.state.trial.should_prune():
                raise optuna.TrialPruned()

        # ---------------------------------------------------------------------
        # 5) Optional detailed breakdown for classification-capable models
        # ---------------------------------------------------------------------
        epoch_idx = engine.state.epoch
        if args.model in ["sscnn", "rsscnn"]:
            # Computes extra diagnostic breakdowns (e.g., per-class accuracy) for the
            # classification output branch; used for interpretability/debugging.
            _compute_class_breakdown(args, net, val_loader, device, "Validation", epoch_idx)

        # ---------------------------------------------------------------------
        # 6) Optional model-specific evaluation behavior
        # ---------------------------------------------------------------------
        # Some backbones/wrappers expose a partial_eval method to evaluate in a
        # reduced or specialized mode (e.g., freezing stochastic components).
        if hasattr(net, "partial_eval"):
            net.partial_eval()

        # Restore training mode so the next epoch uses dropout, etc.
        net.train()

        # ---------------------------------------------------------------------
        # 7) Assemble a consolidated metrics dictionary for logging
        # ---------------------------------------------------------------------
        metrics = {
            "accuracy_train": engine.state.metrics.get("acc"),
            "accuracy_validation": evaluator.state.metrics.get("acc"),
            "accuracy_test": evaluator_test.state.metrics.get("acc"),
        
            "loss_train": engine.state.metrics.get("loss"),
            "loss_validation": evaluator.state.metrics.get("loss"),
            "loss_test": evaluator_test.state.metrics.get("loss"),
        
            "time": f"{timer() - start_training:.3f}",
            "epoch": engine.state.epoch,
            "iteration": engine.state.iteration,
        
            "max_accuracy_validation": training_state["best_val_acc"],
            "max_accuracy_train": training_state["train_acc_at_best_val"],
            "max_accuracy_test": training_state["test_acc_at_best_val"],
            "epoch_best_val": training_state["epoch_best_val"],
        }
        
        if args.model == "rsscnn":
            metrics.update(
                {
                    "loss_kl_train": engine.state.metrics.get("loss_kl") or 0.0,
                    "loss_kl_validation": evaluator.state.metrics.get("loss_kl") or 0.0,
                    "loss_kl_test": evaluator_test.state.metrics.get("loss_kl") or 0.0,
        
                    "c_accuracy_train": engine.state.metrics.get("c_acc"),
                    "c_accuracy_validation": evaluator.state.metrics.get("c_acc"),
                    "c_accuracy_test": evaluator_test.state.metrics.get("c_acc"),
                }
            )
        
        # ---------------------------------------------------------------------
        # 8) Early stopping (optional) based on a chosen validation metric
        # ---------------------------------------------------------------------
        if early_stopper is not None:
            # Select which validation signal to monitor. Defaults to validation accuracy
            # if the requested metric is not available.
            monitor_name = getattr(args, "early_stop_metric", "accuracy_validation")

            # Map of possible monitored signals computed at validation time.
            available = {
                "accuracy_validation": float(evaluator.state.metrics.get("acc", 0.0)),
                "loss_validation": float(evaluator.state.metrics.get("loss", 0.0)),
            }
            if args.model == "rsscnn":
                available["c_accuracy_validation"] = float(evaluator.state.metrics.get("c_acc", 0.0))

            if monitor_name not in available:
                monitor_name = "accuracy_validation"

            current_value = float(available[monitor_name])

            # Update early stopper state and determine whether training should terminate.
            should_stop, _ = early_stopper.update(engine.state.epoch, current_value)

            # Log early stopping diagnostics for transparency and post-hoc analysis.
            #metrics.update(
            #    {
            #        "early_stop/metric": monitor_name,
            #        "early_stop/value": current_value,
            #        "early_stop/best": None if early_stopper.best is None else float(early_stopper.best),
            #        "early_stop/best_epoch": None if early_stopper.best_epoch is None else int(early_stopper.best_epoch),
            #        "early_stop/bad_epochs": int(early_stopper.bad_epochs),
            #    }
            #)

            if should_stop:
                # Human-readable stop reason for logs and W&B summary.
                stop_reason = (
                    f"Early stopping: no improvement in '{monitor_name}' "
                    f"for {early_stopper.patience} epoch(s)."
                )

                # Persist early-stop information into W&B run summary for quick inspection.
                if args.log_wandb and wandb.run is not None:
                    wandb.summary["early_stopped"] = True
                    wandb.summary["early_stop_reason"] = stop_reason
                    wandb.summary["early_stop_metric"] = monitor_name
                    wandb.summary["early_stop_mode"] = getattr(args, "early_stop_mode", "max")
                    wandb.summary["early_stop_patience"] = getattr(args, "early_stop_patience", 3)
                    wandb.summary["early_stop_min_delta"] = getattr(args, "early_stop_min_delta", 0.0)
                    wandb.summary["early_stop_best"] = None if early_stopper.best is None else float(early_stopper.best)
                    wandb.summary["early_stop_best_epoch"] = None if early_stopper.best_epoch is None else int(early_stopper.best_epoch)
                    wandb.summary["early_stop_stopped_epoch"] = int(engine.state.epoch)

                # Stop the trainer engine cleanly at the end of this epoch.
                engine.terminate()

        # ---------------------------------------------------------------------
        # 9) Emit metrics to the project's logger (console / file / W&B, depending on args)
        # ---------------------------------------------------------------------
        log(args, metrics)

    return log_validation_results

# --------------------------------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------------------------------

def train(
    device,
    net,
    train_loader,
    val_loader,
    test_loader,
    args,
    logger,
    trial=None,
    train_df=None,
    val_df=None,
    test_df=None,
    train_tfms=None,
    eval_tfms=None,
):
    """
    Main Ignite-based training entrypoint.

    What it does:
      - Configure training utilities and shared state
      - Move model to device and configure model-level switches (e.g., gaze backprop)
      - Build optimizer/scheduler (transformer-aware, optional backbone freezing)
      - Build Ignite engines (trainer / evaluator / evaluator_test) and attach metrics/handlers
      - Run training
      - Reload best-validation weights and run final validation/test evaluation
      - Return a scalar objective for sweeps (smoothed validation accuracy when available)
    """

    # -----------------------------
    # Training utilities
    # -----------------------------
    accum_steps = max(1, getattr(args, "k", 1))              # gradient accumulation factor
    grad_clip = float(getattr(args, "grad_clip", 0.0))      # global norm clipping threshold

    early_stopper = None
    if getattr(args, "early_stop", False):
        early_stopper = EarlyStopper(
            patience=getattr(args, "early_stop_patience", 3),
            min_delta=getattr(args, "early_stop_min_delta", 0.0),
            mode=getattr(args, "early_stop_mode", "max"),
            start_epoch=getattr(args, "early_stop_start_epoch", 1),
        )

    # Shared state used by handlers (best tracking, history, best weights)
    training_state: Dict[str, Any] = {
        "best_val_acc": 0.0,
        "train_acc_at_best_val": float("-inf"),
        "test_acc_at_best_val": float("-inf"),
        "epoch_best_val": None,
        "val_acc_history": [],
        "best_state_dict": None,  # populated when a new best val epoch is reached
    }

    # -----------------------------
    # Model setup
    # -----------------------------
    net = net.to(device)  # critical: move parameters/buffers to target device

    # DataParallel-safe config accessor (attributes like `.transformer` live under `.module`)
    net_cfg = net.module if isinstance(net, torch.nn.DataParallel) else net
    is_transformer = hasattr(net_cfg, "transformer")

    gaze_mode, _use_gaze_any, use_gaze_kl, _use_gaze_inj = _resolve_gaze_flags(args)
    
    attn_w = float(getattr(args, "attn_w", 0.0) or 0.0)
    
    # Attention-map gradient retention is only needed when KL actually contributes to the objective.
    # In off/guide modes KL is diagnostic only (w_kl_eff=0), so attention grads are not required.
    _set_gaze_bp(net, enabled=bool(use_gaze_kl and (attn_w > 0.0)))


    # Split trainable parameters into: head / GII / backbone
    head_params, gii_params, backbone_params = _split_parameters(net)
    
    # Build optimizer and optional freeze controller (single source of truth for param groups)
    optimizer, optimizer_info, freeze_ctl = build_optimizer(
        args=args,
        net=net,
        is_transformer=is_transformer,
        head_params=head_params,
        gii_params=gii_params,
        backbone_params=backbone_params,
    )


    # Build scheduler (accounts for accumulation and train loader length)
    scheduler, scheduler_type = _build_scheduler(
        args, optimizer, accum_steps, len(train_loader), args.base_lr
    )

    # Print run configuration summary (debug/repro)
    print_run_plan(
        args,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_loader=train_loader,
        val_loader=val_loader,
        train_tfms=train_tfms,
        eval_tfms=eval_tfms,
        model=net_cfg,
        optimizer=optimizer,
        optimizer_info=optimizer_info,
        scheduler=scheduler,
    )

    # -----------------------------
    # Ignite engines
    # -----------------------------
    trainer = Engine(
        _make_train_step(
            args=args,
            device=device,
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_type=scheduler_type,
            accum_steps=accum_steps,
            grad_clip=grad_clip,
            logger=logger,
            trial=trial,
        )
    )

    evaluator = Engine(_make_inference_step(args, device, net))
    evaluator_test = Engine(_make_inference_step(args, device, net))

    trainer.state.trial = trial  # makes trial available to handlers

    _attach_metrics([trainer, evaluator, evaluator_test], args, device)          # fills engine.state.metrics
    _attach_epoch_end_step(trainer, optimizer, accum_steps)                      # handles accumulation bookkeeping

    # -----------------------------
    # Epoch-end evaluation handler
    # -----------------------------
    start_training = timer()

    validation_handler = _make_validation_handler(
        args=args,
        net=net,
        trainer=trainer,
        evaluator=evaluator,
        evaluator_test=evaluator_test,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_type=scheduler_type,
        val_loader=val_loader,
        test_loader=test_loader,
        early_stopper=early_stopper,
        start_training=start_training,
        training_state=training_state,   # contains best_state_dict storage
        device=device,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, validation_handler)

    # -----------------------------
    # Checkpointing
    # -----------------------------
    run_name = getattr(getattr(wandb, "run", None), "name", "no_wandb")
    """
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        ModelCheckpoint(
            args.model_dir,
            f"{args.model}_{args.backbone}",
            n_saved=10,
            create_dir=True,
            require_empty=False,
            score_function=lambda e: e.state.metrics["val_acc"],
            global_step_transform=lambda *_: trainer.state.epoch,
        ),
        {"model": net},
    )
    """
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        ModelCheckpoint(
            args.model_dir,
            f"{run_name}_best",
            n_saved=1,
            create_dir=True,
            require_empty=False,
            score_function=lambda e: e.state.metrics["val_acc"],
            global_step_transform=lambda *_: trainer.state.epoch,
        ),
        {"model": net},
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        ModelCheckpoint(
            args.model_dir,
            f"{run_name}_last",
            n_saved=1,
            create_dir=True,
            require_empty=False,
            global_step_transform=lambda *_: trainer.state.epoch,
        ),
        {"model": net},
    )

    # -----------------------------
    # Resume support
    # -----------------------------
    if getattr(args, "resume", False):

        def _set_start_epoch(engine):
            engine.state.epoch = args.epoch

        def _set_max_epoch(engine):
            engine.state.max_epochs = args.max_epochs

        for eng in [trainer, evaluator, evaluator_test]:
            eng.add_event_handler(Events.STARTED, _set_start_epoch)
            eng.add_event_handler(Events.STARTED, _set_max_epoch)

    optimizer.zero_grad(set_to_none=True)  # clean gradient state at start
    
    # Optional backbone unfreeze controller (epoch-gated)
    @trainer.on(Events.EPOCH_STARTED)
    def _maybe_unfreeze_backbone(engine):
        if freeze_ctl is None:
            return
        epoch0 = int(engine.state.epoch) - 1  # Ignite epochs are 1-based
        did_unfreeze = freeze_ctl.maybe_unfreeze(epoch0)
        if did_unfreeze:
            print(f"[optimizer] Backbone unfrozen at Ignite epoch {engine.state.epoch} (epoch0={epoch0})")

    # ------------------------------------------------------------------------------------------------
    # Run training loop
    # ------------------------------------------------------------------------------------------------
    trainer.run(train_loader, max_epochs=args.max_epochs)
    
    """
    # ------------------------------------------------------------------------------------------------
    # Final evaluation using best-validation weights (not last epoch)
    # ------------------------------------------------------------------------------------------------
    best_sd = training_state.get("best_state_dict")
    
    if isinstance(best_sd, dict) and len(best_sd) > 0:
        model_to_load = net.module if hasattr(net, "module") else net
        model_to_load.load_state_dict(best_sd, strict=True)
    
        net.eval()
        evaluator.run(val_loader)
        evaluator_test.run(test_loader)
        net.train()
    
        training_state["final_best_val_acc"] = float(evaluator.state.metrics.get("acc", 0.0))
        training_state["final_best_test_acc"] = float(evaluator_test.state.metrics.get("acc", 0.0))
    else:
        training_state["final_best_val_acc"] = float(training_state.get("best_val_acc", 0.0))
        training_state["final_best_test_acc"] = float(training_state.get("test_acc_at_best_val", 0.0))
    
    # ------------------------------------------------------------------------------------------------
    # LOG final evaluation using BEST validation weights (so console/W&B end on epoch_best_val weights)
    # ------------------------------------------------------------------------------------------------

    final_best_metrics = {
        # Make it visually “after last epoch” in plots
        "epoch": int(trainer.state.epoch) + 1,
        "iteration": int(trainer.state.iteration) + 1,
        "time": f"{timer() - start_training:.3f}",

        # Report BEST-weights results as the final point
        "accuracy_train": float(training_state.get("train_acc_at_best_val", 0.0)),
        "accuracy_validation": float(training_state.get("final_best_val_acc", 0.0)),
        "accuracy_test": float(training_state.get("final_best_test_acc", 0.0)),

        # Keep the “best tracking” keys consistent
        "max_accuracy_validation": float(training_state.get("best_val_acc", 0.0)),
        "max_accuracy_train": float(training_state.get("train_acc_at_best_val", 0.0)),
        "max_accuracy_test": float(training_state.get("test_acc_at_best_val", 0.0)),
        "epoch_best_val": int(training_state.get("epoch_best_val") or -1),

        # Optional flag to disambiguate in W&B
        "final_best_eval": 1,
    }
    log(args, final_best_metrics)
    """
    # ------------------------------------------------------------------------------------------------
    # W&B finalization (after final eval so final metrics can be logged)
    # ------------------------------------------------------------------------------------------------
    if getattr(args, "log_wandb", False) and wandb.run is not None:
        wandb.finish()

    # ------------------------------------------------------------------------------------------------
    # Objective computation
    # ------------------------------------------------------------------------------------------------
    val_acc_history: List[float] = training_state["val_acc_history"]
    best_val_acc = float(training_state["best_val_acc"])
    
    if len(val_acc_history) >= 3:
        final_val_acc = sum(val_acc_history[-3:]) / 3.0
    elif len(val_acc_history) > 0:
        final_val_acc = sum(val_acc_history) / len(val_acc_history)
    else:
        final_val_acc = best_val_acc
    
    if trial is not None:
        trial.set_user_attr("best_val_acc", float(best_val_acc))
        trial.set_user_attr("final_val_acc", float(final_val_acc))
    
    return final_val_acc
