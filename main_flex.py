# mvte/main_flex.py

import argparse
import logging
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
import json
import time

import torch
import yaml

from data_topology import build_topology_from_triples
from model import MVTEModel
from training import train_model, move_topology_to_device
from utils import (
    set_random_seed,
    setup_logging,
    load_triples_from_file,
    build_id_mappings,
    encode_triples,
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


def main() -> None:
    start_time = time.time()
    """entry point: load config, data, topology, model, and run training."""
    args = parse_args()
    cfg = load_config(args.config)

    # derive a run/dataset name for subdirectories
    run_name = get_run_name_from_config(cfg)

    run_root = Path(cfg.get("output", {}).get("save_dir", "runs"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # e.g. checkpoints/FB15k-237/20251202_141522
    run_dir = run_root / run_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    import csv
    metrics_csv_path = run_dir / "metrics.csv"
    metrics_history = []  # keep in memory too, for summary.json at the end


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
    source = data_cfg.get("source", "files")

    if source == "pykeen":        # PyKEEN path

        try:
            from pykeen.datasets import FB15k237, get_dataset
        except ImportError as e:
            raise ImportError(
                "data.source is set to 'pykeen' but pykeen is not installed. "
                "Install it with `pip install pykeen`."
            ) from e

        dataset_name = data_cfg.get("dataset", "FB15k237")
        if dataset_name == "FB15k237":
            ds = FB15k237()
        else:
            ds = get_dataset(dataset=dataset_name)

        logger.info(
            "loaded PyKEEN dataset %s: %d entities, %d relations",
            dataset_name,
            ds.num_entities,
            ds.num_relations,
        )
        # mapped_triples are already integer IDs [N, 3] (h, r, t)
        train_triples = ds.training.mapped_triples.to(device)
        valid_triples = (
            ds.validation.mapped_triples.to(device)
            if ds.validation is not None
            else None
        )
        test_triples = (
            ds.testing.mapped_triples.to(device)
            if ds.testing is not None
            else None
        )

        num_entities = ds.num_entities
        num_relations = ds.num_relations

        # real label to id maps from the dataset (used in checkpoints/testing)

        entity2id = dict(ds.entity_to_id)
        relation2id = dict(ds.relation_to_id)

    else:        # file path loading
        train_path = Path(data_cfg["train_path"])
        valid_path = data_cfg.get("valid_path", None)
        test_path = data_cfg.get("test_path", None)

        if not train_path.exists():
            raise FileNotFoundError(f"train file not found: {train_path}")

        # load raw triples (string ids)
        logger.info("loading training triples from %s", train_path)
        train_triples_raw = load_triples_from_file(train_path)

        valid_triples_raw = None
        test_triples_raw = None
        # in case of validation / test paths not provided (just train)
        if valid_path is not None:
            valid_triples_raw = load_triples_from_file(Path(valid_path))
            logger.info("loaded validation triples from %s", valid_path)
        if test_path is not None:
            test_triples_raw = load_triples_from_file(Path(test_path))
            logger.info("loaded test triples from %s", test_path)

        # build entity / relation id mappings
        entity2id, relation2id = build_id_mappings(
            train_triples_raw,
            valid_triples_raw,
            test_triples_raw,
        )

        num_entities = len(entity2id)
        num_relations = len(relation2id)
        logger.info("num entities: %d | num relations: %d", num_entities, num_relations)

        # encode triples to integer tensors
        train_triples = encode_triples(train_triples_raw, entity2id, relation2id)
        train_triples = train_triples.to(device)

        valid_triples = None
        test_triples = None
        if valid_triples_raw is not None:
            valid_triples = encode_triples(
                valid_triples_raw,
                entity2id,
                relation2id,
            ).to(device)
        if test_triples_raw is not None:
            test_triples = encode_triples(
                test_triples_raw,
                entity2id,
                relation2id,
            ).to(device)
    
    # build union of triples for filtered evaluation
    # (kept on CPU because evaluation internally moves to CPU to build a set)
    all_triples = train_triples.cpu()
    if valid_triples is not None:
        all_triples = torch.cat([all_triples, valid_triples.cpu()], dim=0)
    if test_triples is not None:
        all_triples = torch.cat([all_triples, test_triples.cpu()], dim=0)

    # prepare output directories
    output_cfg = cfg.get("output", {})
    base_output_dir = output_cfg.get("save_dir", None)
    out_dir: Path | None = None

    if base_output_dir is not None:
        # e.g. checkpoints/<run_name>/<timestamp>/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(base_output_dir) / run_name / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("output will be saved to %s", out_dir)
    

    if out_dir is not None:
        encoded_dir = out_dir / "encoded_splits"
        encoded_dir.mkdir(parents=True, exist_ok=True)

        # Save encoded splits (to cpu tensors because it's safer)
        torch.save(train_triples.cpu(), encoded_dir / "train_ids.pt")
        if valid_triples is not None:
            torch.save(valid_triples.cpu(), encoded_dir / "valid_ids.pt")
        if test_triples is not None:
            torch.save(test_triples.cpu(),  encoded_dir / "test_ids.pt")


        logger.info("saved encoded splits to %s", encoded_dir)


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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
        logger.info(
            (
                "validation epoch %d | loss=%.4f | MR=%.1f | MRR=%.4f | "
                "Hits@1=%.4f | Hits@3=%.4f | Hits@10=%.4f"
            ),
            epoch,
            mean_loss,
            metrics["MR"],
            metrics["MRR"],
            metrics.get("Hits@1", float("nan")),
            metrics.get("Hits@3", float("nan")),
            metrics.get("Hits@10", float("nan")),
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
        }
        metrics_history.append(record)
        # append to CSV (create header on first write)
        write_header = not metrics_csv_path.exists()
        with metrics_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(record)

        

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


    # optional saving

    if out_dir is not None:
        # save fused vector embeddings and model checkpoint
        fused_path = out_dir / "fused_entity_embeddings.pt"
        torch.save(
            {
                "V_fused": V_fused_cpu,
                "entity2id": entity2id,  # to align rows with entity ids
            },
            fused_path,
        )
        logger.info("saved fused entity embeddings to %s", fused_path)
        # saving of full checkpoint
        model_path = out_dir / "mvte_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "entity2id": entity2id,
                "relation2id": relation2id,
                "config": cfg,
                "V_fused": V_fused_cpu,

            },
            model_path,
        )
        logger.info("saved model checkpoint to %s", model_path)

        with (out_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        logger.info("saved config to %s", out_dir / "config_used.yaml")


    logger.info("done")


    end_time = time.time()
    elapsed_sec = end_time - start_time

    
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
