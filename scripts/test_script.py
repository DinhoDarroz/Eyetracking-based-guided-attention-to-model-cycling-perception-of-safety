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

            if gaze_l is None:
                # Legacy datasets: create dummy tensors
                gaze_l = torch.zeros((label_r.size(0), 14, 14), device=device)
                gaze_r = torch.zeros((label_r.size(0), 14, 14), device=device)
                has_eye_mask = torch.zeros((label_r.size(0),), dtype=torch.bool, device=device)
            else:
                gaze_l = gaze_l.to(device)
                gaze_r = gaze_r.to(device)
                has_eye_mask = has_eye.to(device)

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
            forward_dict = net(input_left, input_right)
            loss = compute_loss(args, forward_dict, labels)

            # ----------------------------
            # 3) Prepare outputs for metrics
            # ----------------------------
            # Also prepare numpy arrays for saving to disk
            input_left_name = data["image_l_name"]
            input_right_name = data["image_r_name"]

            if args.model == "rcnn":
                rank_left_t = forward_dict["left"]["output"].view(-1)
                rank_right_t = forward_dict["right"]["output"].view(-1)

                rank_left = rank_left_t.detach().cpu().numpy()
                rank_right = rank_right_t.detach().cpu().numpy()

                forward_pass = {
                    "rank_left": rank_left,
                    "rank_right": rank_right,
                }

                returnable_dict = {
                    "loss": loss.item(),
                    "rank_left": rank_left_t,
                    "rank_right": rank_right_t,
                    "label": label_r,
                }

            elif args.model == "sscnn":
                logits_t = forward_dict["logits"]["output"]
                logits_np = logits_t.detach().cpu().numpy()

                if args.ties:
                    forward_pass = {
                        "logits_l": logits_np[:, 0],
                        "logits_0": logits_np[:, 1],
                        "logits_r": logits_np[:, 2],
                    }
                else:
                    forward_pass = {
                        "logits_l": logits_np[:, 0],
                        "logits_r": logits_np[:, 1],
                    }

                returnable_dict = {
                    "loss": loss.item(),
                    "logits": logits_t,
                    "label": label_c,
                }

            elif args.model == "rsscnn":
                rank_left_t = forward_dict["left"]["output"].view(-1)
                rank_right_t = forward_dict["right"]["output"].view(-1)
                logits_t = forward_dict["logits"]["output"]

                rank_left = rank_left_t.detach().cpu().numpy()
                rank_right = rank_right_t.detach().cpu().numpy()
                logits_np = logits_t.detach().cpu().numpy()

                if args.ties:
                    forward_pass = {
                        "rank_left": rank_left,
                        "rank_right": rank_right,
                        "logits_l": logits_np[:, 0],
                        "logits_0": logits_np[:, 1],
                        "logits_r": logits_np[:, 2],
                    }
                else:
                    forward_pass = {
                        "rank_left": rank_left,
                        "rank_right": rank_right,
                        "logits_l": logits_np[:, 0],
                        "logits_r": logits_np[:, 1],
                    }

                returnable_dict = {
                    "loss": loss.item(),
                    "rank_left": rank_left_t,
                    "rank_right": rank_right_t,
                    "logits": logits_t,
                    "label_r": label_r,
                    "label_c": label_c,
                }

            else:
                raise ValueError(f"Unknown model type: {args.model}")

            # ----------------------------
            # 4) Save per-batch outputs
            # ----------------------------
            output_dict = {
                "image_left": input_left_name,
                "image_right": input_right_name,
                "label_r": data["score_r"],
                "label_c": data["score_c"],
            }
            output_dict.update(forward_pass)

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
            Accuracy(
                output_transform=lambda x: (x["logits"], x["label"])
            ).attach(engine, "acc")

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

            Accuracy(
                output_transform=lambda x: (x["logits"], x["label_c"])
            ).attach(engine, "c_acc")

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
