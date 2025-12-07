# mvte/main_flex.py

import argparse
import logging
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
import json
import time
import csv


import torch
import yaml

from data_topology import build_topology_from_triples
from model import MVTEModel
from training import train_model, move_topology_to_device
from utils import (
    set_random_seed,
    setup_logging,
    get_device_from_config,
    get_run_name_from_config,
)

from evaluation import evaluate_link_prediction


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="train mvte topology-aware kg embeddings"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to yaml config file",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    """load yaml config file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def _load_prepared_dataset(
    data_cfg: Dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, Dict[str, int], Dict[str, int]]:
    """
    Load integer-encoded splits and mappings from a prepared dataset directory.

    Expects:
        data.prepared_dir: path like "data/FB15k-237"

    And inside that directory:
        - train_ids.pt
        - valid_ids.pt
        - test_ids.pt 
        - entity2id.pt
        - relation2id.pt
    """
    if "prepared_dir" not in data_cfg:
        raise ValueError(
            "config.data.prepared_dir is not set.\n"
            "You are now in the 'prepared dataset' workflow. "
            "Run data_prep.py first to create a prepared dataset directory, "
            "then point config.data.prepared_dir to it."
        )

    dataset_dir = Path(data_cfg["prepared_dir"]) # will be resolved relative to the current working directory
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Prepared dataset directory not found: {dataset_dir}"
        )

    logger.info("loading prepared dataset from %s", dataset_dir)

    train_path = dataset_dir / "train_ids.pt"
    valid_path = dataset_dir / "valid_ids.pt"
    test_path = dataset_dir / "test_ids.pt"
    ent_map_path = dataset_dir / "entity2id.pt"
    rel_map_path = dataset_dir / "relation2id.pt"

    if not train_path.exists():
        raise FileNotFoundError(f"train_ids.pt not found in {dataset_dir}")
    if not ent_map_path.exists() or not rel_map_path.exists():
        raise FileNotFoundError(
            f"entity2id.pt / relation2id.pt not found in {dataset_dir}"
        )

    # integer triples [N, 3]
    train_triples = torch.load(train_path).long().to(device)
    valid_triples = torch.load(valid_path).long().to(device) if valid_path.exists() else None
    test_triples = torch.load(test_path).long().to(device) if test_path.exists() else None

    entity2id: Dict[str, int] = torch.load(ent_map_path)
    relation2id: Dict[str, int] = torch.load(rel_map_path)

    logger.info(
        "loaded splits: train=%d, valid=%s, test=%s",
        train_triples.shape[0],
        "None" if valid_triples is None else valid_triples.shape[0],
        "None" if test_triples is None else test_triples.shape[0],
    )
    logger.info(
        "num entities: %d | num relations: %d",
        len(entity2id),
        len(relation2id),
    )

    return train_triples, valid_triples, test_triples, entity2id, relation2id



def main() -> None:
    """entry point: load config, data, topology, model, and run training."""
    start_time = time.time()
    args = parse_args()
    cfg = load_config(args.config)

    # derive a run/dataset name for subdirectories
    run_name = get_run_name_from_config(cfg)

    run_root = Path(cfg.get("output", {}).get("save_dir", "runs"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # e.g. runs/FB15k-237/20251202_141522
    run_dir = run_root / run_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv_path = run_dir / "metrics.csv"
    metrics_history: list[dict[str, Any]] = []  # for summary.json at the end


    # logging 
    log_cfg = cfg.get("logging", {})
    log_level = log_cfg.get("level", "INFO")
    base_log_dir = log_cfg.get("log_dir", None)
    log_dir = None
    if base_log_dir is not None:
        log_dir = str(Path(base_log_dir) / run_name / timestamp)


    setup_logging(log_level=log_level, log_dir=log_dir)

    logger.info("loaded config from %s", args.config)

    # seeding 
    seed = cfg.get("training", {}).get("seed", 42)
    set_random_seed(seed)
    logger.info("random seed set to %d", seed)

    # device 
    device = get_device_from_config(cfg.get("training", {}).get("device", "auto"))
    logger.info("using device: %s", device)

    # data loading: either from local files or a PyKEEN dataset
    data_cfg = cfg.get("data", {})
    (
        train_triples,
        valid_triples,
        test_triples,
        entity2id,
        relation2id,
    ) = _load_prepared_dataset(data_cfg=data_cfg, device=device)

    num_entities = len(entity2id)
    num_relations = len(relation2id)

    # build union of triples for filtered evaluation
    # (kept on CPU because evaluation internally moves to CPU to build a set)
    all_triples = train_triples.cpu()
    if valid_triples is not None:
        all_triples = torch.cat([all_triples, valid_triples.cpu()], dim=0)
    if test_triples is not None:
        all_triples = torch.cat([all_triples, test_triples.cpu()], dim=0)


    # topology construction (from train triples)
    topo_cfg = cfg.get("topology", {})
    max_tri_per_ent = topo_cfg.get("max_triangles_per_entity", None)
    max_tet_per_ent = topo_cfg.get("max_tetras_per_entity", None)

    logger.info("building topology from training triples")
    topo = build_topology_from_triples(
        num_entities=num_entities,
        triples=train_triples.cpu(),  # topology is cpu-side; moved in training
        max_triangles_per_entity=max_tri_per_ent,
        max_tetras_per_entity=max_tet_per_ent,
    )
    logger.info(
        "topology built: %d triangles, %d tetras",
        topo.triangles.size(0),
        topo.tetras.size(0),
    )

    # model instantiation
    model_cfg = cfg.get("model", {})
    embedding_dim = model_cfg.get("embedding_dim", 200)
    tri_hidden_dim = model_cfg.get("tri_hidden_dim", embedding_dim)
    tet_hidden_dim = model_cfg.get("tet_hidden_dim", embedding_dim)
    dropout = model_cfg.get("dropout", 0.0)
    gamma = model_cfg.get("gamma", 12.0)

    # evaluation config
    eval_cfg = cfg.get("evaluation", {})
    eval_batch_size_entities = eval_cfg.get("batch_size_entities", 1024)
    eval_filtered = eval_cfg.get("filtered", True)
    eval_hits_ks = eval_cfg.get("hits_ks", [1, 3, 10])
    eval_every = eval_cfg.get("eval_every", 1)  # run every epoch by default

    model = MVTEModel(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        tri_hidden_dim=tri_hidden_dim,
        tet_hidden_dim=tet_hidden_dim,
        dropout=dropout,
        gamma=gamma,
    )
    model.fusion_mode = model_cfg.get("fusion_mode", "equal")  # "learned" | "topo_only" | "equal"
    model.to(device)

    # optimizer and training cfg
    train_cfg = cfg.get("training", {})
    lr = train_cfg.get("lr", 1e-3)
    num_epochs = train_cfg.get("num_epochs", 100)
    batch_size = train_cfg.get("batch_size", 1024)
    num_negatives = train_cfg.get("num_negatives", 128)
    adv_temp = train_cfg.get("adversarial_temperature", 1.0)
    negative_mode = train_cfg.get("negative_mode", "both")
    log_interval = train_cfg.get("log_interval", 100)
    checkpoint_every = train_cfg.get("checkpoint_every", 0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # model summary logging
    # ------------------------------------------------------------------
    w1_init, w2_init = model.get_fusion_weights()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        (
            "starting training of MVTEModel\n"
            "  num_entities      = %d\n"
            "  num_relations     = %d\n"
            "  embedding_dim     = %d\n"
            "  tri_hidden_dim    = %d\n"
            "  tet_hidden_dim    = %d\n"
            "  dropout           = %.3f\n"
            "  gamma (fixed)     = %.4f\n"
            "  fusion_alpha      = [%+.4f, %+.4f]\n"
            "  initial w1, w2    = %.4f, %.4f\n"
            "  total_parameters  = %d\n"
            "  fusion_mode      = %s"
        ),
        num_entities,
        num_relations,
        embedding_dim,
        tri_hidden_dim,
        tet_hidden_dim,
        dropout,
        float(model.gamma.item()),
        float(model.fusion_alpha[0].item()),
        float(model.fusion_alpha[1].item()),
        float(w1_init),
        float(w2_init),
        num_params,
        model.fusion_mode,
    )
    # ------------------------------------------------------------------
    # training setup summary logging
    # ------------------------------------------------------------------
    logger.info(
        (
            "training configuration:\n"
            "  seed                  = %d\n"
            "  device                = %s\n"
            "  learning_rate         = %.6f\n"
            "  num_epochs            = %d\n"
            "  batch_size            = %d\n"
            "  num_negatives         = %d\n"
            "  adversarial_temp      = %.4f\n"
            "  negative_mode         = %s\n"
            "  log_interval          = %d\n"
            "  checkpoint_every      = %d"
        ),
        seed,
        str(device),
        lr,
        num_epochs,
        batch_size,
        num_negatives,
        adv_temp,
        negative_mode,
        log_interval,
        checkpoint_every,
    )

    logger.info("========== STARTING TRAINING RUN ==========")



    def epoch_callback(epoch: int, mean_loss: float) -> None:
        """callback run at the end of each epoch to perform evaluation."""
        if valid_triples is None:
            return
        if eval_every > 1 and (epoch % eval_every) != 0:
            return
        logger.info("running link prediction evaluation at epoch %d", epoch)
        # This uses evaluation.py. If V is None, it will:
        #   - move topo to device,
        #   - compute V_fused once via model.get_entity_views(topo_on_device),
        #   - reuse it inside rank computation.
        max_eval = eval_cfg.get("max_eval_triples", None) # limit eval triples for speed
        eval_triples = valid_triples
        if max_eval is not None and eval_triples.size(0) > max_eval:
            # simple random subset
            idx = torch.randperm(eval_triples.size(0))[:max_eval]
            eval_triples = eval_triples[idx]

        metrics = evaluate_link_prediction(
            model=model,
            topo=topo,
            eval_triples=eval_triples, # use possibly subsetted valid triples
            all_triples=all_triples,
            device=device,
            V=None,  # let evaluation compute fresh V on current parameters
            batch_size_entities=eval_batch_size_entities,
            filtered=eval_filtered,
            hits_ks=eval_hits_ks,
        )
        w1, w2 = model.get_fusion_weights()
        logger.info(
            (
                "validation epoch %d | loss=%.4f | MR=%.1f | MRR=%.4f | "
                "Hits@1=%.4f | Hits@3=%.4f | Hits@10=%.4f"
                "Fusion weights: w1=%.4f, w2=%.4f"
            ),
            epoch,
            mean_loss,
            metrics["MR"],
            metrics["MRR"],
            metrics.get("Hits@1", float("nan")),
            metrics.get("Hits@3", float("nan")),
            metrics.get("Hits@10", float("nan")),
            float(w1),
            float(w2),
        )

        # NEW: store a structured record
        record = {
            "epoch": epoch,
            "train_loss": float(mean_loss),
            "valid_MR": float(metrics["MR"]),
            "valid_MRR": float(metrics["MRR"]),
            "valid_Hits@1": float(metrics.get("Hits@1", float("nan"))),
            "valid_Hits@3": float(metrics.get("Hits@3", float("nan"))),
            "valid_Hits@10": float(metrics.get("Hits@10", float("nan"))),
            "w1": float(w1),
            "w2": float(w2),
        }
        metrics_history.append(record)
        # append to CSV (create header on first write)
        write_header = not metrics_csv_path.exists()
        with metrics_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(record)
        # optional checkpointing
        if checkpoint_every > 0 and (epoch % checkpoint_every) == 0:
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )
            logger.info("saved periodic checkpoint to %s", ckpt_path)

        

    # training
    logger.info("starting training for %d epochs", num_epochs)
    train_model(
        model=model,
        topo=topo,
        train_triples=train_triples,
        optimizer=optimizer,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_negatives=num_negatives,
        adversarial_temperature=adv_temp,
        device=device,
        negative_mode=negative_mode,
        log_interval=log_interval,
        epoch_callback=epoch_callback,
    )

    # compute fused entity embeddings V once at the end of training

    topo_on_device = move_topology_to_device(topo, device)
    with torch.no_grad():
        _, _, V_fused = model.get_entity_views(topo_on_device)

    # store on CPU for saving / later evaluation
    V_fused_cpu = V_fused.detach().cpu()
    logger.info("computed fused entity embeddings V with shape %s", tuple(V_fused_cpu.shape))

    end_time = time.time()
    elapsed_sec = end_time - start_time
    logger.info("training completed in %.2f seconds", elapsed_sec)

    # optional saving
    # save fused vector embeddings and model checkpoint
    fused_path = run_dir / "fused_entity_embeddings.pt"
    torch.save(
        {
            "V_fused": V_fused_cpu,
            "entity2id": entity2id,  # to align rows with entity ids
        },
        fused_path,
    )
    logger.info("saved fused entity embeddings to %s", fused_path)
    # saving of full checkpoint
    model_path = run_dir / "mvte_model.pt"
    w1, w2 = model.get_fusion_weights()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "entity2id": entity2id,
            "relation2id": relation2id,
            "config": cfg,
            "V_fused": V_fused_cpu,
            "fusion_weights": {
                "w1": float(w1),
                "w2": float(w2),
            }
        },
        model_path,
    )
    logger.info("saved model checkpoint to %s", model_path)

    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    logger.info("saved config to %s", run_dir / "config_used.yaml")


    logger.info("done")


    
    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_seconds": elapsed_sec,
        "dataset": cfg.get("data", {}),
        "model": cfg.get("model", {}),
        "training": cfg.get("training", {}),
        "evaluation": cfg.get("evaluation", {}),
    }
    # topology details if available
    if topo is not None:
        summary["topology"] = {
            "num_entities": topo.num_entities,
            "num_triangles": int(topo.triangles.size(0)),
            "num_tetras": int(topo.tetras.size(0)),
        }
    
    # best validation epoch by MRR (if we have any)
    if metrics_history:
        best = max(metrics_history, key=lambda r: r["valid_MRR"])
        summary["best_validation"] = {
            "epoch": best["epoch"],
            "MRR": best["valid_MRR"],
            "Hits@10": best["valid_Hits@10"],
        }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)





if __name__ == "__main__":
    main()
