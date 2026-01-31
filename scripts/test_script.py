# scripts/test_script.py

import os
from os import path
from glob import glob

import pandas as pd
from tqdm import tqdm

import torch
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Accuracy

from utils.log import log
from utils.accuracy import RankAccuracy, RankAccuracy_withMargin, RankAccuracy_ties
from utils.losses import compute_loss


def test(device, net, dataloader, args, logger=None):
    """
    Evaluate a trained model on a comparisons DataLoader.

    - Works for rcnn, sscnn, rsscnn
    - Supports ties / full_accuracy flags
    - Uses the new compute_loss (incl. gaze KL if enabled)
    - Saves per-batch outputs into 'outputs/' and an aggregated
      dataframe into 'outputs/saved/'.

    Args:
        device: torch.device
        net: trained model
        dataloader: DataLoader over ComparisonsDataset
        args: argparse.Namespace with at least:
              model, ties, full_accuracy,
              ranking_margin, ranking_margin_ties, attn_w, notes,
              checkpoint (or load_model)
        logger: optional logging.Logger
    """

    os.makedirs("outputs", exist_ok=True)
    os.makedirs(path.join("outputs", "saved"), exist_ok=True)

    # Figure out a base name for output files
    ckpt_name = getattr(args, "checkpoint", None) or getattr(args, "load_model", "model")
    ckpt_base = path.basename(ckpt_name)

    # --------------------------------------------------------------------------------------- #
    # INFERENCE STEP
    # --------------------------------------------------------------------------------------- #
    def inference(engine, data):
        with torch.no_grad():
            # ----------------------------
            # 1) Move data to device
            # ----------------------------
            input_left = data["image_l"].to(device)
            input_right = data["image_r"].to(device)

            # Ranking & classification labels
            label_r = data["score_r"].to(device).float()
            label_c = data["score_c"].to(device).long()

            # Gaze tensors + mask
            gaze_l = data.get("gaze_l", None)
            gaze_r = data.get("gaze_r", None)
            has_eye = data.get("has_eyetracker", None)

            if gaze_l is None or gaze_r is None or has_eye is None:
                # Legacy datasets: create dummy tensors
                ms = int(getattr(args, "gaze_map_size_int", 14))
                gaze_l = torch.zeros((label_r.size(0), ms, ms), device=device)
                gaze_r = torch.zeros((label_r.size(0), ms, ms), device=device)

                has_eye_mask = torch.zeros((label_r.size(0),), dtype=torch.bool, device=device)
            else:
                gaze_l = gaze_l.to(device)
                gaze_r = gaze_r.to(device)
                has_eye_mask = has_eye.to(device)

                # New dataset can return [B,1,H,W] (or dummy [B,1,1,1]); losses expect [B,H,W]
                if gaze_l.ndim == 4 and gaze_l.size(1) == 1:
                    gaze_l = gaze_l.squeeze(1)
                if gaze_r.ndim == 4 and gaze_r.size(1) == 1:
                    gaze_r = gaze_r.squeeze(1)

            labels = {
                "label_r": label_r,
                "label_c": label_c,
                "gaze_l": gaze_l,
                "gaze_r": gaze_r,
                "has_eye_mask": has_eye_mask,
            }

            # ----------------------------
            # 2) Forward + loss
            # ----------------------------
            # Resolve gaze mode and whether injection is active
            gaze_mode = str(getattr(args, "gaze_mode", getattr(args, "gaze", "off"))).lower().strip()
            if gaze_mode not in ("off", "align", "guide", "align+guide"):
                gaze_mode = "off"
            
            use_gaze_inj = (gaze_mode in ("guide", "align+guide"))
            
            # Detect transformer wrapper (DataParallel-safe)
            net_cfg = net.module if isinstance(net, torch.nn.DataParallel) else net
            is_transformer = hasattr(net_cfg, "transformer")
            
            # Guided forward only for transformer models
            if use_gaze_inj and is_transformer:
                forward_dict = net(input_left, input_right, gaze_l, gaze_r, has_eye_mask)
            else:
                forward_dict = net(input_left, input_right)


            # compute_loss returns total loss + optional parts (class / rank / KL) when return_parts=True
            loss_t, parts = compute_loss(args, forward_dict, labels, return_parts=True)

            # Safe float extraction (parts keys may not exist depending on model / flags)
            def _pf(key: str) -> float:
                v = parts.get(key, None) if isinstance(parts, dict) else None
                if v is None:
                    return 0.0
                if torch.is_tensor(v):
                    return float(v.item())
                try:
                    return float(v)
                except Exception:
                    return 0.0

            loss_total = float(loss_t.item())
            loss_class = _pf("loss_class")
            loss_rank_combo = _pf("loss_rank_combo")
            loss_kl = _pf("loss_kl")
            loss_kl_weighted = _pf("loss_kl_weighted")

            # ----------------------------
            # 3) Prepare outputs for metrics
            # ----------------------------
            # Also prepare numpy arrays for saving to disk
            input_left_name = data["image_l_name"]
            input_right_name = data["image_r_name"]

            # ----------------------------
            # 4) Save per-batch outputs
            # ----------------------------
            output_dict = {
                "image_left": input_left_name,
                "image_right": input_right_name,
                "label_r": data["score_r"],
                "label_c": data["score_c"],
                "loss_total": loss_total,
                "loss_class": loss_class,
                "loss_rank": loss_rank_combo,
                "loss_kl": loss_kl,
                "loss_kl_weighted": loss_kl_weighted,
            }

            # ----------------------------
            # 5) Return values for metrics (ignite)
            # ----------------------------
            if args.model == "rcnn":
                rank_left_t = forward_dict["left"]["output"].view(-1)
                rank_right_t = forward_dict["right"]["output"].view(-1)

                rank_left = rank_left_t.detach().cpu().numpy()
                rank_right = rank_right_t.detach().cpu().numpy()

                output_dict.update(
                    {
                        "rank_left": rank_left,
                        "rank_right": rank_right,
                    }
                )

                returnable_dict = {
                    "loss": loss_total,
                    "rank_left": rank_left_t,
                    "rank_right": rank_right_t,
                    "label": label_r,
                    "loss_rank_combo": loss_rank_combo,
                    "loss_kl": loss_kl,
                    "loss_kl_weighted": loss_kl_weighted,
                }

            elif args.model == "sscnn":
                logits_t = forward_dict["logits"]["output"]
                logits_np = logits_t.detach().cpu().numpy()

                if args.ties:
                    output_dict.update(
                        {
                            "logits_l": logits_np[:, 0],
                            "logits_0": logits_np[:, 1],
                            "logits_r": logits_np[:, 2],
                        }
                    )
                else:
                    output_dict.update(
                        {
                            "logits_l": logits_np[:, 0],
                            "logits_r": logits_np[:, 1],
                        }
                    )

                returnable_dict = {
                    "loss": loss_total,
                    "logits": logits_t,
                    "label": label_c,
                    "loss_class": loss_class,
                }

            elif args.model == "rsscnn":
                rank_left_t = forward_dict["left"]["output"].view(-1)
                rank_right_t = forward_dict["right"]["output"].view(-1)
                logits_t = forward_dict["logits"]["output"]

                rank_left = rank_left_t.detach().cpu().numpy()
                rank_right = rank_right_t.detach().cpu().numpy()
                logits_np = logits_t.detach().cpu().numpy()

                output_dict.update(
                    {
                        "rank_left": rank_left,
                        "rank_right": rank_right,
                    }
                )

                if args.ties:
                    output_dict.update(
                        {
                            "logits_l": logits_np[:, 0],
                            "logits_0": logits_np[:, 1],
                            "logits_r": logits_np[:, 2],
                        }
                    )
                else:
                    output_dict.update(
                        {
                            "logits_l": logits_np[:, 0],
                            "logits_r": logits_np[:, 1],
                        }
                    )

                returnable_dict = {
                    "loss": loss_total,
                    "rank_left": rank_left_t,
                    "rank_right": rank_right_t,
                    "logits": logits_t,
                    "label_r": label_r,
                    "label_c": label_c,
                    "loss_class": loss_class,
                    "loss_rank_combo": loss_rank_combo,
                    "loss_kl": loss_kl,
                    "loss_kl_weighted": loss_kl_weighted,
                }

            else:
                raise ValueError(f"Unknown model type: {args.model}")

            df_batch = pd.DataFrame(output_dict)
            batch_fname = f"{ckpt_base}_{engine.state.iteration}.pkl"
            df_batch.to_pickle(path.join("outputs", batch_fname))

            pbar.update(1)

            return returnable_dict

    # --------------------------------------------------------------------------------------- #
    # BUILD EVALUATOR
    # --------------------------------------------------------------------------------------- #
    net = net.to(device)
    net.eval()

    evaluator = Engine(inference)

    # Logging at the end of the evaluation (single pass)
    @evaluator.on(Events.COMPLETED)
    def log_validation_results(evaluator):
        metrics = {
            "accuracy_validation": evaluator.state.metrics["acc"],
            "loss_validation": evaluator.state.metrics["loss"],
            "epoch": evaluator.state.epoch,
            "iteration": evaluator.state.iteration,
        }

        # When gaze != off, show loss breakdown + KL (if compute_loss provides it)
        if getattr(args, "gaze", "off") != "off":
            if "loss_class" in evaluator.state.metrics:
                metrics["loss_class_validation"] = evaluator.state.metrics["loss_class"]
            if "loss_rank_combo" in evaluator.state.metrics:
                metrics["loss_rank_validation"] = evaluator.state.metrics["loss_rank_combo"]
            if "loss_kl" in evaluator.state.metrics:
                metrics["loss_kl_validation"] = evaluator.state.metrics["loss_kl"]
            if "loss_kl_weighted" in evaluator.state.metrics:
                metrics["loss_kl_weighted_validation"] = evaluator.state.metrics["loss_kl_weighted"]

        if args.full_accuracy and args.ties and args.model != "sscnn":
            metrics["accuracy_validation_ties"] = evaluator.state.metrics["acc_ties"]

        if args.model == "rsscnn":
            metrics["c_accuracy_validation"] = evaluator.state.metrics["c_acc"]

        log(args, metrics)

    # --------------------------------------------------------------------------------------- #
    # METRICS
    # --------------------------------------------------------------------------------------- #
    for engine in [evaluator]:
        # Always log average loss across all batches
        RunningAverage(
            output_transform=lambda x: x["loss"],
            device=device,
        ).attach(engine, "loss")

        # Optional: component losses (logged only if present in returnable_dict)
        RunningAverage(
            output_transform=lambda x: x.get("loss_class", 0.0),
            device=device,
        ).attach(engine, "loss_class")
        RunningAverage(
            output_transform=lambda x: x.get("loss_rank_combo", 0.0),
            device=device,
        ).attach(engine, "loss_rank_combo")
        RunningAverage(
            output_transform=lambda x: x.get("loss_kl", 0.0),
            device=device,
        ).attach(engine, "loss_kl")
        RunningAverage(
            output_transform=lambda x: x.get("loss_kl_weighted", 0.0),
            device=device,
        ).attach(engine, "loss_kl_weighted")

        # Ranking only
        if args.model == "rcnn":
            if args.full_accuracy:
                RankAccuracy_withMargin(
                    output_transform=lambda x: (
                        x["rank_left"],
                        x["rank_right"],
                        x["label"],
                        args.ranking_margin,
                    ),
                    device=device,
                ).attach(engine, "acc")
                if args.ties:
                    RankAccuracy_ties(
                        output_transform=lambda x: (
                            x["rank_left"],
                            x["rank_right"],
                            x["label"],
                            args.ranking_margin,
                        ),
                        device=device,
                    ).attach(engine, "acc_ties")
            else:
                RankAccuracy(
                    output_transform=lambda x: (
                        x["rank_left"],
                        x["rank_right"],
                        x["label"],
                    ),
                    device=device,
                ).attach(engine, "acc")

        # SSCNN (classification only)
        elif args.model == "sscnn":
            Accuracy(output_transform=lambda x: (x["logits"], x["label"])).attach(engine, "acc")

        # RSSCNN (ranking + classification)
        elif args.model == "rsscnn":
            if args.full_accuracy:
                RankAccuracy_withMargin(
                    output_transform=lambda x: (
                        x["rank_left"],
                        x["rank_right"],
                        x["label_r"],
                        args.ranking_margin,
                    ),
                    device=device,
                ).attach(engine, "acc")
                if args.ties:
                    RankAccuracy_ties(
                        output_transform=lambda x: (
                            x["rank_left"],
                            x["rank_right"],
                            x["label_r"],
                            args.ranking_margin,
                        ),
                        device=device,
                    ).attach(engine, "acc_ties")
            else:
                RankAccuracy(
                    output_transform=lambda x: (
                        x["rank_left"],
                        x["rank_right"],
                        x["label_r"],
                    ),
                    device=device,
                ).attach(engine, "acc")

            Accuracy(output_transform=lambda x: (x["logits"], x["label_c"])).attach(engine, "c_acc")

        else:
            raise Exception(f"Model type unknown: {args.model}")

    # --------------------------------------------------------------------------------------- #
    # RUN EVALUATION
    # --------------------------------------------------------------------------------------- #
    pbar = tqdm(total=len(dataloader))
    evaluator.run(dataloader)
    pbar.close()

    # --------------------------------------------------------------------------------------- #
    # MERGE BATCH RESULTS
    # --------------------------------------------------------------------------------------- #
    batch_result_files = glob(path.join("outputs", f"{ckpt_base}_*.pkl"))
    batch_results = [pd.read_pickle(f) for f in batch_result_files]

    # Delete temporary files
    for f in batch_result_files:
        os.remove(f)

    if len(batch_results) > 0:
        global_df = pd.concat(batch_results, axis=0)
    else:
        global_df = pd.DataFrame()

    out_name = f"{getattr(args, 'notes', '')}_{ckpt_base}_results.pkl"
    out_name = out_name.lstrip("_")  # avoid leading underscore when notes=""

    global_df.to_pickle(path.join("outputs", "saved", out_name))
    print(global_df)
    print(global_df.shape)

    return global_df
