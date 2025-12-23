import sys
import torch.nn as nn
import torch.optim as optim
import torch
import time
import wandb
import math
import optuna
import os

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Accuracy
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from random import randint
from timeit import default_timer as timer
import wandb
import inspect
from torch.optim import lr_scheduler

from utils.log import log
from utils.losses import compute_loss
from utils.accuracy import RankAccuracy, RankAccuracy_withMargin, RankAccuracy_ties

class EarlyStopper:
    """
    Simple epoch-level early stopping.

    Monitors a scalar metric once per epoch (after validation is computed).
    Stops training after `patience` epochs without improvement.

    - mode="max": improvement means metric > best + min_delta
    - mode="min": improvement means metric < best - min_delta
    """

    def __init__(self, patience=3, min_delta=0.0, mode="max", start_epoch=1):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.start_epoch = int(start_epoch)

        self.best = None
        self.best_epoch = None
        self.bad_epochs = 0

    def _is_improvement(self, current):
        if self.best is None:
            return True

        if self.mode == "max":
            return current > (self.best + self.min_delta)
        else:  # "min"
            return current < (self.best - self.min_delta)

    def update(self, epoch, current):
        """
        Returns (should_stop: bool, improved: bool)
        """
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

def train(device, net, dataloader, val_loader, test_loader, args, logger, trial=None):
    """
    Main training loop.

    High-level flow:
      STEP 0:  Basic training configuration (grad accumulation, clipping, tracking metrics)
      STEP 1:  Define the training step (update) used by Ignite's trainer
      STEP 2:  Define the inference step used by validation & test engines
      STEP 3:  Build optimizer (+ param groups) and LR scheduler
      STEP 4:  Create Ignite engines (trainer, evaluator, evaluator_test)
      STEP 5:  Attach metrics (loss, acc, etc.) to engines
      STEP 6:  Define epoch-end hooks (validation, logging, Optuna pruning, confusion)
      STEP 7:  Set up checkpoints and resume behavior
      STEP 8:  Run training loop and compute final Optuna objective
    """

    # =============================================================================================== #
    # STEP 0: BASIC TRAINING CONFIGURATION
    # =============================================================================================== #

    # Gradient accumulation: number of mini-batches per optimizer step
    accum_steps = max(1, getattr(args, "k", 1))
    
    # Gradient clipping: max norm (<=0 means disabled)
    grad_clip = getattr(args, "grad_clip", 0.0)
    
    # Track best validation accuracy across epochs (for Optuna / monitoring)
    best_val_acc = 0.0
    
    # -----------------------------------------------------------------------------------
    # Early stopping setup (epoch-level; uses validation metric)
    # -----------------------------------------------------------------------------------
    early_stopper = None
    if getattr(args, "early_stop", False):
        early_stopper = EarlyStopper(
            patience=getattr(args, "early_stop_patience", 3),
            min_delta=getattr(args, "early_stop_min_delta", 0.0),
            mode=getattr(args, "early_stop_mode", "max"),
            start_epoch=getattr(args, "early_stop_start_epoch", 1),
        )

    # Track per-epoch validation accuracies (for final Optuna objective)
    val_acc_history = []

    # =============================================================================================== #
    # STEP 1: UPDATE (TRAIN STEP)
    # =============================================================================================== #

    """
    Single training iteration ("train step") used by the Ignite trainer engine.

    For each batch, this function:
    - Moves input images and labels to the target device.
    - Applies gaze / eye-tracker masking if requested.
    - Runs a forward pass through the network (CNN or Transformer wrapper).
    - Computes the appropriate loss:
        * ranking loss      (rcnn / rsscnn)
        * classification loss (sscnn / rsscnn)
        * or both, depending on args.model.
    - Handles NaN checks for the loss (fail-fast debugging).
    - Implements gradient accumulation:
        * Scales the loss by 1/accum_steps.
        * Calls loss.backward() every iteration.
        * Calls optimizer.step() and scheduler.step() only every `accum_steps` iterations.
    - Optionally applies gradient clipping before optimizer.step().
    - Returns a dictionary with tensors needed for Ignite metrics:
        * loss, rank_left / rank_right, classification logits, labels
          (depending on the selected model).
    """

    def update(engine, data):
        
        if os.path.exists("SKIP_TRIAL"):
            print("[USER REQUEST] SKIPPING THIS RUN NOW.")
            os.remove("SKIP_TRIAL")
        
            # If running under Optuna, prune cleanly
            if trial is not None:
                raise optuna.TrialPruned()
        
            # Otherwise (W&B sweep / normal run), terminate Ignite cleanly
            engine.terminate()
            return {"skipped": True}


        if logger:
            start = timer()

        # -------------------------------
        # 1) Move inputs & labels to GPU
        # -------------------------------
        # Load input data
        input_left, input_right = data['image_l'], data['image_r']
        input_left, input_right = input_left.to(device), input_right.to(device)

        # Load label data
        label_r, label_c = data['score_r'], data['score_c']
        label_r, label_c = label_r.to(device), label_c.to(device)
        # Ranking label → float, classification label → long
        label_r = label_r.float()
        label_c = label_c.long()

        # Gaze + eye tracker mask
        gaze_l, gaze_r = data['gaze_l'], data['gaze_r']
        gaze_l, gaze_r = gaze_l.to(device), gaze_r.to(device)
        has_eye_mask = data['has_eyetracker'].to(device)

        labels = {
            'label_r': label_r,
            'label_c': label_c,
            'gaze_l': gaze_l,
            'gaze_r': gaze_r,
            'has_eye_mask': has_eye_mask
        }

        # NOTE: we DO NOT call optimizer.zero_grad() here.
        # Gradients will accumulate across `accum_steps` mini-batches.

        # -------------------------------
        # 2) Forward + loss
        # -------------------------------
        # Forward pass the training sample
        forward_dict = net(input_left, input_right)

        # Compute loss (now includes attention KL if args.attn_w > 0)
        loss = compute_loss(args, forward_dict, labels)

        # Keep an unscaled copy for logging / metrics
        raw_loss = loss

        # --- DEBUG: catch NaN early ---
        if torch.isnan(loss):
            # label_r: -1, 0, +1 for ranking
            n_ties = (label_r == 0).sum().item()
            n_nonties = (label_r != 0).sum().item()
            print(
                f"[NaN DETECTED] epoch={engine.state.epoch} "
                f"iter={engine.state.iteration} "
                f"batch_size={label_r.size(0)}, "
                f"n_nonties={n_nonties}, n_ties={n_ties}"
            )
            raise ValueError("NaN loss, stopping for debug.")

        # --------------------------------------------------
        # 3) Gradient accumulation logic
        # --------------------------------------------------

        # Scale loss so that after `accum_steps` backward() calls,
        # the total gradient equals the big-batch gradient.
        loss = loss / accum_steps

        # Backpropagate: gradients are ACCUMULATED in param.grad
        loss.backward()

        # Only update weights every `accum_steps` iterations
        if engine.state.iteration % accum_steps == 0:

            # Optional gradient clipping (helps stabilize ViT training)
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            if scheduler and scheduler_type != "plateau":
                scheduler.step()


        if logger:
            logger.info(f'TRAIN_STEP, {timer() - start:.4f}')

        # -------------------------------
        # 4) Prepare outputs for metrics
        # -------------------------------

        # Ranking Model Only
        if args.model == 'rcnn':
            return {
                'loss': raw_loss.item(),
                'rank_left': forward_dict['left']['output'],
                'rank_right': forward_dict['right']['output'],
                'label': label_r
            }

        # Classification Model Only
        elif args.model == 'sscnn':
            return {
                'loss': raw_loss.item(),
                'logits': forward_dict['logits']['output'],
                'label': label_c.long(),
            }

        # Classification + Ranking Model
        elif args.model == 'rsscnn':
            return {
                'loss': raw_loss.item(),
                'rank_left': forward_dict['left']['output'],
                'rank_right': forward_dict['right']['output'],
                'logits': forward_dict['logits']['output'],
                'label_r': label_r,
                'label_c': label_c
            }

    # =============================================================================================== #
    # STEP 2: INFERENCE (VAL / TEST STEP)
    # =============================================================================================== #

    """
    Single evaluation (inference) step used by the validation and test Ignite engines.

    This function:
    - Disables gradient computation with torch.no_grad().
    - Moves inputs and labels to the device.
    - Applies the same gaze / eye-tracker masking as in the training step.
    - Runs a forward pass to compute model outputs.
    - Computes the loss with the same loss functions as in training
      (ranking and/or classification).
    - Returns a dictionary containing loss, logits, ranking scores,
      and labels so that Ignite metrics can be computed on val/test sets.
    """
    def inference(engine, data):
        with torch.no_grad():
            # -------------------------------
            # 1) Move inputs & labels to GPU
            # -------------------------------
            # Load input data
            input_left, input_right = data['image_l'], data['image_r']
            input_left, input_right = input_left.to(device), input_right.to(device)

            # Load label data
            label_r, label_c = data['score_r'], data['score_c']
            label_r, label_c = label_r.to(device), label_c.to(device)
            label_r = label_r.float()
            label_c = label_c.long()

            # Gaze + eye tracker mask       
            gaze_l, gaze_r = data['gaze_l'], data['gaze_r']
            gaze_l, gaze_r = gaze_l.to(device), gaze_r.to(device)
            has_eye_mask = data['has_eyetracker'].to(device)

            labels = {
                'label_r': label_r,
                'label_c': label_c,
                'gaze_l': gaze_l,
                'gaze_r': gaze_r,
                'has_eye_mask': has_eye_mask
            }

            # -------------------------------
            # 2) Forward + loss
            # -------------------------------
            # Forward pass the sample
            forward_dict = net(input_left, input_right)

            loss = compute_loss(args, forward_dict, labels)

            # -------------------------------
            # 3) Prepare outputs for metrics
            # -------------------------------

            # Ranking Model Only
            if args.model == 'rcnn':
                return {
                    'loss': loss.item(),
                    'rank_left': forward_dict['left']['output'],
                    'rank_right': forward_dict['right']['output'],
                    'label': label_r
                }

            # Classification Model Only
            elif args.model == 'sscnn':
                return {
                    'loss': loss.item(),
                    'logits': forward_dict['logits']['output'],
                    'label': label_c.long(),
                }

            # Classification + Ranking Model
            elif args.model == 'rsscnn':
                return {
                    'loss': loss.item(),
                    'rank_left': forward_dict['left']['output'],
                    'rank_right': forward_dict['right']['output'],
                    'logits': forward_dict['logits']['output'],
                    'label_r': label_r,
                    'label_c': label_c
                }

    # =============================================================================================== #
    # STEP 3: MODEL & OPTIMIZER (PARAM GROUPS + SCHEDULER)
    # =============================================================================================== #
    net = net.to(device)

    # Detect whether this is a Transformer wrapper (has .transformer) or a CNN
    is_transformer = hasattr(net, "transformer")

    # ---- Split params: backbone vs heads (works for both CNN & Transformer) ----
    head_params, backbone_params = [], []
    for n, p in net.named_parameters():
        if any(k in n for k in ["rank_fc", "cross_fc"]):  # MLP heads
            head_params.append((n, p))
        else:
            backbone_params.append((n, p))

    # ---- Separate weight decay for norms and biases (used only for Transformer) ----
    def separate_decay(params):
        decay, no_decay = [], []
        for n, p in params:
            if p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower() or "ln" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        return decay, no_decay

    base_lr = args.base_lr
    weight_decay = args.weight_decay

    # ------------------------------ TRANSFORMER OPTIMIZER (AdamW) ------------------------------ #
    if is_transformer:
        print("[Optimizer] Using AdamW with parameter groups (Transformer backbone).")

        if not args.finetune:
            # ----------------------------------------------------------------------------
            # FEATURE EXTRACTOR MODE: freeze backbone, train ONLY the MLP heads
            # ----------------------------------------------------------------------------
            for n, p in backbone_params:
                p.requires_grad = False

            head_decay, head_no_decay = separate_decay(head_params)

            optimizer = optim.AdamW(
                [
                    # Head (MLP) parameters WITH weight decay
                    {
                        "params": head_decay,
                        "lr": base_lr,
                        "weight_decay": weight_decay
                    },

                    # Head (MLP) parameters WITHOUT weight decay
                    {
                        "params": head_no_decay,
                        "lr": base_lr,
                        "weight_decay": 0.0
                    },
                ],
                betas=(0.9, 0.999),
                eps=1e-8,
            )

        else:
            # ----------------------------------------------------------------------------
            # FINETUNE MODE: train backbone + heads, smaller LR on backbone
            # ----------------------------------------------------------------------------
            head_decay, head_no_decay = separate_decay(head_params)
            backbone_decay, backbone_no_decay = separate_decay(backbone_params)

            backbone_lr = base_lr * args.backbone_lr_scale

            optimizer = optim.AdamW(
                [
                    # Backbone WITH weight decay
                    {
                        "params": backbone_decay,
                        "lr": backbone_lr,
                        "weight_decay": weight_decay
                    },

                    # Backbone WITHOUT weight decay
                    {
                        "params": backbone_no_decay,
                        "lr": backbone_lr,
                        "weight_decay": 0.0
                    },

                    # Head (MLP) WITH weight decay
                    {
                        "params": head_decay,
                        "lr": base_lr,
                        "weight_decay": weight_decay
                    },

                    # Head (MLP) WITHOUT weight decay
                    {
                        "params": head_no_decay,
                        "lr": base_lr,
                        "weight_decay": 0.0
                    },
                ],
                betas=(0.9, 0.999),
                eps=1e-8,
            )

    # ------------------------------ CNN OPTIMIZER (Adam) ------------------------------ #
    else:
        print("[Optimizer] Using Adam (CNN backbone).")

        if not args.finetune:
            # Freeze backbone, train ONLY heads
            for n, p in backbone_params:
                p.requires_grad = False

            cnn_head_params = [p for (_, p) in head_params if p.requires_grad]

            optimizer = optim.Adam(
                cnn_head_params,
                lr=base_lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
            )
        else:
            # Finetune backbone + heads together
            cnn_all_params = [p for (_, p) in head_params + backbone_params if p.requires_grad]

            optimizer = optim.Adam(
                cnn_all_params,
                lr=base_lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
            )

    # ---- LR schedule configuration (all schedulers are optimizer-step-based) ----
    steps_per_epoch = len(dataloader)                     
    eff_steps_per_epoch = math.ceil(steps_per_epoch / accum_steps)
    total_iters = args.max_epochs * eff_steps_per_epoch   
    
    scheduler_type = args.scheduler
    scheduler = None
    
    if scheduler_type == "none":
        scheduler = None
    
    elif scheduler_type == "warmup_cosine":
        warmup_frac = max(0.0, min(1.0, args.warmup_frac))
        warmup_iters = int(warmup_frac * total_iters)
        eta_min = args.eta_min
    
        def lr_lambda(step: int):
            if warmup_iters > 0 and step < warmup_iters:
                return float(step) / float(max(1, warmup_iters))
            if total_iters == warmup_iters:
                return eta_min / base_lr
    
            progress = min(
                1.0,
                max(0.0, (step - warmup_iters) / float(max(1, total_iters - warmup_iters)))
            )
            eta_min_factor = eta_min / base_lr
            return eta_min_factor + (1 - eta_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
    
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_iters,
            eta_min=args.eta_min,
        )
    
    elif scheduler_type == "onecycle":
        max_lrs = [pg["lr"] for pg in optimizer.param_groups]
        warmup_frac = max(0.0, min(1.0, args.warmup_frac))
        pct_start = warmup_frac if warmup_frac > 0 else 0.3
    
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_iters,
            pct_start=pct_start,
            anneal_strategy="cos",
            cycle_momentum=False,
        )
    
    elif scheduler_type == "warm_restarts":
        # NEW: CosineAnnealingWarmRestarts
        #
        # T_0 = number of optimizer steps before the first restart
        # T_mult = restart cycle length multiplier
        #
        # Important: Use optimizer-step stepping, not epoch-level stepping
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.T_0,
            T_mult=args.T_mult,
            eta_min=args.eta_min,
        )
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",                 # we monitor validation accuracy
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.plateau_min_lr,
            #verbose=True,
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # =============================================================================================== #
    # STEP 4: ENGINES (TRAIN / VAL / TEST)
    # =============================================================================================== #
    trainer = Engine(update)
    evaluator = Engine(inference)
    evaluator_test = Engine(inference)

    # If an epoch ends with leftover accumulated gradients, use them
    @trainer.on(Events.EPOCH_COMPLETED)
    def step_on_epoch_end(engine):
        # If the last optimizer.step() wasn't called because we didn't hit
        # an exact multiple of accum_steps, do a final step here.
        if engine.state.iteration % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            """
            current_val_acc = float(evaluator.state.metrics['acc'])
            # Step LR scheduler (Plateau ONLY)
            if scheduler and scheduler_type == "plateau":
                scheduler.step(current_val_acc)
            """
    # =============================================================================================== #
    # STEP 5: HELPER FOR CLASSIFICATION BREAKDOWN (CONFUSION-LIKE STATS)
    # =============================================================================================== #
    def compute_class_breakdown(loader, split_name, epoch_idx, print_output=True):
        """
        Computes a confusion matrix adapted to:
        - ties=True  → 3 classes  (0=left, 1=tie, 2=right)
        - ties=False → 2 classes  (0=left, 1=right)
    
        If print_output=False → do NOT print anything (useful for test set).
        """
    
        # Only relevant for models that have a classification head
        if args.model not in ['sscnn', 'rsscnn']:
            return
    
        # --------------------------
        # Determine number of classes
        # --------------------------
        if args.ties:
            num_classes = 3
            class_names = ["left", "tie", "right"]
        else:
            num_classes = 2
            class_names = ["left", "right"]
    
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
        net.eval()
        with torch.no_grad():
            for batch in loader:
                input_left = batch['image_l'].to(device)
                input_right = batch['image_r'].to(device)
    
                # score_c already contains correct 0/1 or 0/1/2 labels depending on ties
                label_c = batch['score_c'].to(device).long()
    
                forward_dict = net(input_left, input_right)
                logits = forward_dict['logits']['output']  # [B, num_classes]
                preds = torch.argmax(logits, dim=1)
    
                for t, p in zip(label_c.view(-1), preds.view(-1)):
                    t = int(t.item())
                    p = int(p.item())
                    if 0 <= t < num_classes and 0 <= p < num_classes:
                        confusion[t, p] += 1
    
        # --------------------------
        # Printing logic
        # --------------------------
        if not print_output:
            return confusion  # silent mode
    
        print(f"\n[Epoch {epoch_idx}] {split_name} classification breakdown (true x pred)")
        print("Rows = true class, Cols = predicted class")
        print("Class mapping:", {i: name for i, name in enumerate(class_names)})
        print(confusion.cpu().numpy())
    
        # Per-class breakdown
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
    

    # =============================================================================================== #
    # STEP 6: METRICS ATTACHMENT (LOSS / ACCURACY)
    # =============================================================================================== #
    for engine in [trainer, evaluator, evaluator_test]:

        # ---------------------------------------------------------
        # RCNN (ranking only)
        # ---------------------------------------------------------
        if args.model == 'rcnn':
            RunningAverage(
                output_transform=lambda x: x['loss'],
                device=device
            ).attach(engine, 'loss')

            # Non-tie accuracy
            if args.full_accuracy:
                RankAccuracy_withMargin(
                    output_transform=lambda x: (
                        x['rank_left'], x['rank_right'], x['label'], args.ranking_margin
                    ),
                    device=device
                ).attach(engine, 'acc')
            else:
                RankAccuracy(
                    output_transform=lambda x: (
                        x['rank_left'], x['rank_right'], x['label']
                    ),
                    device=device
                ).attach(engine, 'acc')
            """
            # Tie accuracy
            if args.ties:
                RankAccuracy_ties(
                    output_transform=lambda x: (
                        x['rank_left'], x['rank_right'], x['label'], args.ranking_margin
                    ),
                    device=device
                ).attach(engine, 'acc_ties')
            """
        # ---------------------------------------------------------
        # SSCNN (classification only)
        # ---------------------------------------------------------
        elif args.model == 'sscnn':
            RunningAverage(
                output_transform=lambda x: x['loss'],
                device=device
            ).attach(engine, 'loss')

            Accuracy(
                output_transform=lambda x: (x['logits'], x['label'])
            ).attach(engine, 'acc')

        # ---------------------------------------------------------
        # RSSCNN (ranking + classification)
        # ---------------------------------------------------------
        elif args.model == 'rsscnn':
            RunningAverage(
                output_transform=lambda x: x['loss'],
                device=device
            ).attach(engine, 'loss')

            # Ranking accuracy for non-ties
            if args.full_accuracy:
                RankAccuracy_withMargin(
                    output_transform=lambda x: (
                        x['rank_left'], x['rank_right'], x['label_r'], args.ranking_margin
                    ),
                    device=device
                ).attach(engine, 'acc')
            else:
                RankAccuracy(
                    output_transform=lambda x: (
                        x['rank_left'], x['rank_right'], x['label_r']
                    ),
                    device=device
                ).attach(engine, 'acc')
            """
            # Tie accuracy
            if args.ties:
                RankAccuracy_ties(
                    output_transform=lambda x: (
                        x['rank_left'], x['rank_right'], x['label_r'], args.ranking_margin
                    ),
                    device=device
                ).attach(engine, 'acc_ties')
            """
            # Classification accuracy
            Accuracy(
                output_transform=lambda x: (x['logits'], x['label_c'])
            ).attach(engine, 'c_acc')



    # =============================================================================================== #
    # STEP 7: EPOCH-END LOGGING, VALIDATION, OPTUNA PRUNING
    # =============================================================================================== #
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        nonlocal best_val_acc, val_acc_history
    
        # Put model in eval mode for evaluation
        net.eval()
    
        # Run evaluators
        evaluator.run(val_loader)
        evaluator_test.run(test_loader)
        trainer.state.metrics['val_acc'] = evaluator.state.metrics['acc']
    
        # Current validation accuracy for THIS epoch
        current_val_acc = float(evaluator.state.metrics['acc'])
    
        # ---- STEP PLATEAU HERE ----
        if scheduler and scheduler_type == "plateau":
            scheduler.step(current_val_acc)
    
            # ===== SANITY CHECK (ADD THIS) =====
            lr_head = optimizer.param_groups[0]["lr"]
            print(
                f"[Plateau sanity] "
                f"epoch={trainer.state.epoch} "
                f"val_acc={current_val_acc:.4f} "
                f"lr={lr_head:.3e}"
            )
            # ================================
    
        # Track history and best-so-far
        val_acc_history.append(current_val_acc)
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
    
        # Report to Optuna + possible pruning
        if trial is not None:
            current_epoch = trainer.state.epoch
    
            # Use BEST-SO-FAR val accuracy for pruning decisions
            trial.report(best_val_acc, step=current_epoch)
    
            # Ask Optuna if we should stop this trial
            if trial.should_prune():
                raise optuna.TrialPruned()
    
        # Classification breakdown
        epoch_idx = trainer.state.epoch
        if args.model in ['sscnn', 'rsscnn']:
            compute_class_breakdown(val_loader, split_name="Validation", epoch_idx=epoch_idx)
            # compute_class_breakdown(test_loader, split_name="Test", epoch_idx=epoch_idx)
    
        # Optional partial eval hook
        if hasattr(net, 'partial_eval'):
            net.partial_eval()
    
        # Switch back to train mode before next epoch
        net.train()
    
        # ---------------------------
        # Build metrics dict FIRST
        # ---------------------------
        metrics = {
            'accuracy_train': trainer.state.metrics['acc'],
            'accuracy_validation': evaluator.state.metrics['acc'],
            'accuracy_test': evaluator_test.state.metrics['acc'],
            'loss_train': trainer.state.metrics['loss'],
            'loss_validation': evaluator.state.metrics['loss'],
            'loss_test': evaluator_test.state.metrics['loss'],
            'time': f'{timer() - start_training:.3f}',
            'epoch': trainer.state.epoch,
            'iteration': trainer.state.iteration,
            'max_accuracy_validation': best_val_acc,
            'max_accuracy_train': 0,
            'max_accuracy_test': 0,
        }
    
        # (Your ties block is currently commented out in your file; keep it as-is if desired)
        """
        if args.ties and ('acc_ties' in trainer.state.metrics):
            if args.model in ['rcnn', 'rsscnn']:
                metrics.update({
                    'accuracy_train_ties': trainer.state.metrics['acc_ties'],
                    'accuracy_validation_ties': evaluator.state.metrics['acc_ties'],
                    'accuracy_test_ties': evaluator_test.state.metrics['acc_ties'],
                })
        """
    
        if args.model == 'rsscnn':
            metrics.update({
                'c_accuracy_train': trainer.state.metrics['c_acc'],
                'c_accuracy_validation': evaluator.state.metrics['c_acc'],
                'c_accuracy_test': evaluator_test.state.metrics['c_acc'],
            })
    
        # ---------------------------
        # Early stopping decision (NOW safe to metrics.update)
        # ---------------------------
        if early_stopper is not None:
            monitor_name = getattr(args, "early_stop_metric", "accuracy_validation")
    
            available = {
                "accuracy_validation": float(evaluator.state.metrics.get("acc", 0.0)),
                "loss_validation": float(evaluator.state.metrics.get("loss", 0.0)),
            }
            if args.model == "rsscnn":
                available["c_accuracy_validation"] = float(evaluator.state.metrics.get("c_acc", 0.0))
    
            if monitor_name not in available:
                monitor_name = "accuracy_validation"
    
            current_monitor_value = float(available[monitor_name])
    
            should_stop, _improved = early_stopper.update(trainer.state.epoch, current_monitor_value)
    
            metrics.update({
                "early_stop/metric": monitor_name,
                "early_stop/value": current_monitor_value,
                "early_stop/best": (None if early_stopper.best is None else float(early_stopper.best)),
                "early_stop/best_epoch": (None if early_stopper.best_epoch is None else int(early_stopper.best_epoch)),
                "early_stop/bad_epochs": int(early_stopper.bad_epochs),
            })
    
            if should_stop:
                stop_reason = (
                    f"Early stopping: no improvement in '{monitor_name}' "
                    f"for {early_stopper.patience} epoch(s)."
                )
    
                if args.log_wandb and wandb.run is not None:
                    wandb.summary["early_stopped"] = True
                    wandb.summary["early_stop_reason"] = stop_reason
                    wandb.summary["early_stop_metric"] = monitor_name
                    wandb.summary["early_stop_mode"] = getattr(args, "early_stop_mode", "max")
                    wandb.summary["early_stop_patience"] = getattr(args, "early_stop_patience", 3)
                    wandb.summary["early_stop_min_delta"] = getattr(args, "early_stop_min_delta", 0.0)
                    wandb.summary["early_stop_best"] = (None if early_stopper.best is None else float(early_stopper.best))
                    wandb.summary["early_stop_best_epoch"] = (None if early_stopper.best_epoch is None else int(early_stopper.best_epoch))
                    wandb.summary["early_stop_stopped_epoch"] = int(trainer.state.epoch)
    
                trainer.terminate()
    
        # Finally, log everything (W&B / console)
        log(args, metrics)


    # =============================================================================================== #
    # STEP 8: (OPTIONAL) PER-ITERATION LOGGING HOOK (CURRENTLY DISABLED)
    # =============================================================================================== #
    """
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_results(trainer):
        if trainer.state.iteration % 175 == 0:
            total_batches = trainer.state.epoch_length
            current_batch = (trainer.state.iteration - 1) % total_batches + 1

            metrics = {
                "batch": f"{current_batch}/{total_batches}",
                "loss_train": trainer.state.metrics["loss"],
                "time": f"{timer() - start_training:.3f}",
                "epoch": trainer.state.epoch,
                "iteration": trainer.state.iteration,
            }
    
        if args.log_wandb:
            if is_transformer:
                if args.finetune:
                    # 4 param groups: [0,1]=backbone, [2,3]=head
                    metrics.update({
                        "lr_backbone": float(optimizer.param_groups[0]["lr"]),
                        "lr_head": float(optimizer.param_groups[2]["lr"]),
                    })
                else:
                    # 2 param groups: [0,1]=head only
                    metrics.update({
                        "lr_head": float(optimizer.param_groups[0]["lr"]),
                    })
            else:
                # CNN: single param group (or simple layout)
                metrics.update({
                    "lr_head": float(optimizer.param_groups[0]["lr"]),
                })
    
        log(args, metrics)
    """

    # =============================================================================================== #
    # STEP 9: CHECKPOINTS (BEST, PER-EPOCH, LAST)
    # =============================================================================================== #
    handler = ModelCheckpoint(
        args.model_dir,
        '{}_{}'.format(args.model, args.backbone),
        n_saved=10,
        create_dir=True,
        require_empty=False,
        score_function=lambda engine: engine.state.metrics['val_acc'],
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {'model': net})

    handler_best = ModelCheckpoint(
        args.model_dir,
        '{}'.format(wandb.run.name),
        n_saved=1,
        create_dir=True,
        require_empty=False,
        score_function=lambda engine: engine.state.metrics['val_acc'],
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler_best, {'model': net})

    handler_last = ModelCheckpoint(
        args.model_dir,
        '{}'.format(wandb.run.name),
        n_saved=1,
        create_dir=True,
        require_empty=False,
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler_last, {'model': net})

    # =============================================================================================== #
    # STEP 10: RESUME TRAINING (OPTIONAL)
    # =============================================================================================== #
    if args.resume:
        def start_epoch(engine):
            engine.state.epoch = args.epoch

        def max_epoch(engine):
            engine.state.max_epochs = args.max_epochs

        trainer.add_event_handler(Events.STARTED, start_epoch)
        evaluator.add_event_handler(Events.STARTED, start_epoch)
        evaluator_test.add_event_handler(Events.STARTED, start_epoch)
        trainer.add_event_handler(Events.STARTED, max_epoch)
        evaluator.add_event_handler(Events.STARTED, max_epoch)
        evaluator_test.add_event_handler(Events.STARTED, max_epoch)

    # =============================================================================================== #
    # STEP 11: RUN TRAINING LOOP
    # =============================================================================================== #
    # Make sure gradients start at zero before the first accumulation
    optimizer.zero_grad()

    if logger:
        start_training = timer()

    try:
        trainer.run(dataloader, max_epochs=args.max_epochs)
    finally:
        # Ensure WandB closes EVEN IF the trial is pruned or crashes
        if args.log_wandb and wandb.run is not None:
            wandb.finish()

    # --------------------------------------------------
    # Final objective for Optuna :
    # use the average of the last 3 validation accuracies
    # to reward stable, strong end-of-training performance.
    # --------------------------------------------------
    if len(val_acc_history) >= 3:
        final_val_acc = sum(val_acc_history[-3:]) / 3.0
    elif len(val_acc_history) > 0:
        final_val_acc = sum(val_acc_history) / len(val_acc_history)
    else:
        final_val_acc = best_val_acc
    
    # Store metrics on the Optuna trial for later inspection/printing
    if trial is not None:
        trial.set_user_attr("best_val_acc", float(best_val_acc))
        trial.set_user_attr("final_val_acc", float(final_val_acc))
    
    return final_val_acc
