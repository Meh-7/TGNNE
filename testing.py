# mvte/testing.py

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
import json

from evaluation import evaluate_link_prediction
from model import MVTEModel
from utils import (
    setup_logging,
    set_random_seed,
    get_device_from_config,
    get_run_name_from_config,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MVTE model on the test split"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file (same one used for training).",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_prepared_dataset(
    data_cfg: Dict[str, Any],
    device: torch.device,
):
    if "prepared_dir" not in data_cfg:
        raise ValueError(
            "config.data.prepared_dir is not set. "
            "Testing relies on datasets created by data_prep.py."
        )

    dataset_dir = Path(data_cfg["prepared_dir"])
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Prepared dataset directory not found: {dataset_dir}")

    train_triples = torch.load(dataset_dir / "train_ids.pt").long().to(device)
    valid_path = dataset_dir / "valid_ids.pt"
    test_path = dataset_dir / "test_ids.pt"

    valid_triples = (
        torch.load(valid_path).long().to(device)
        if valid_path.exists()
        else None
    )
    test_triples = (
        torch.load(test_path).long().to(device)
        if test_path.exists()
        else None
    )

    entity2id = torch.load(dataset_dir / "entity2id.pt")
    relation2id = torch.load(dataset_dir / "relation2id.pt")

    return train_triples, valid_triples, test_triples, entity2id, relation2id


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # --------------------------------------------------
    # Logging & seed
    # --------------------------------------------------
    log_cfg = cfg.get("logging", {})
    log_level = log_cfg.get("level", "INFO")
    base_log_dir = log_cfg.get("log_dir", None)

    run_name = get_run_name_from_config(cfg)
    log_dir = (
        str(Path(base_log_dir) / f"{run_name}_test")
        if base_log_dir is not None
        else None
    )

    setup_logging(log_level=log_level, log_dir=log_dir)
    logger.info("loaded config from %s", args.config)

    seed = 123
    set_random_seed(seed)
    logger.info("random seed set to %d", seed)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    train_cfg = cfg.get("training", {})
    device = get_device_from_config(train_cfg.get("device", "auto"))
    logger.info("using device: %s", device)

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------
    test_cfg = cfg.get("testing", {})
    if "checkpoint_path" not in test_cfg:
        raise ValueError("config.testing.checkpoint_path is not set")

    ckpt_path = Path(test_cfg["checkpoint_path"]).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    logger.info("loading checkpoint from %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    out_dir = ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    V_fused = checkpoint.get("V_fused", None)
    if V_fused is not None:
        logger.info("loaded V_fused with shape %s", tuple(V_fused.shape))
    else:
        logger.info("checkpoint has no V_fused; will recompute")

    # --------------------------------------------------
    # Load prepared dataset
    # --------------------------------------------------
    (
        train_triples,
        valid_triples,
        test_triples,
        entity2id,
        relation2id,
    ) = _load_prepared_dataset(cfg.get("data", {}), device)

    if test_triples is None or test_triples.numel() == 0:
        raise ValueError("No test triples found")

    all_triples = train_triples.cpu()
    if valid_triples is not None:
        all_triples = torch.cat([all_triples, valid_triples.cpu()], dim=0)
    all_triples = torch.cat([all_triples, test_triples.cpu()], dim=0)

    if checkpoint["entity2id"] != entity2id:
        raise ValueError("entity2id mismatch between checkpoint and dataset")
    if checkpoint["relation2id"] != relation2id:
        raise ValueError("relation2id mismatch between checkpoint and dataset")

    # --------------------------------------------------
    # Reconstruct model (SAME LOGIC AS TRAINING)
    # --------------------------------------------------
    ckpt_cfg = checkpoint.get("config", cfg)
    model_cfg = ckpt_cfg.get("model", {})

    num_entities = len(entity2id)
    num_relations = len(relation2id)

    embedding_dim = model_cfg.get("embedding_dim", 500)
    tri_hidden_dim = model_cfg.get("tri_hidden_dim", embedding_dim)
    tet_hidden_dim = model_cfg.get("tet_hidden_dim", embedding_dim)
    dropout = model_cfg.get("dropout", 0.0)
    gamma = model_cfg.get("gamma", 12.0)
    base_scorer = model_cfg.get("base_scorer", "transe")

    logger.info(
        "reconstructing MVTEModel: num_entities=%d | num_relations=%d | "
        "embedding_dim=%d | tri_hidden_dim=%d | tet_hidden_dim=%d | "
        "dropout=%.3f | gamma=%.3f | base_scorer=%s",
        num_entities,
        num_relations,
        embedding_dim,
        tri_hidden_dim,
        tet_hidden_dim,
        dropout,
        gamma,
        base_scorer,
    )

    model = MVTEModel(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=embedding_dim,   # base_dim
        tri_hidden_dim=tri_hidden_dim,
        tet_hidden_dim=tet_hidden_dim,
        dropout=dropout,
        gamma=gamma,
        base_scorer=base_scorer,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # --------------------------------------------------
    # Recompute V_fused if needed
    # --------------------------------------------------
    if V_fused is None:
        logger.info("recomputing V_fused from model")
        with torch.no_grad():
            V_fused = model.get_entity_views(topo=None)

        logger.info("recomputed V_fused with shape %s", tuple(V_fused.shape))

    if V_fused.size(0) != num_entities:
        raise ValueError("V_fused entity count mismatch")

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    eval_cfg = cfg.get("evaluation", {})
    batch_size_entities = eval_cfg.get("batch_size_entities", 1024)
    filtered = eval_cfg.get("filtered", True)
    hits_ks = eval_cfg.get("hits_ks", [1, 3, 10])

    logger.info(
        "starting evaluation: %d test triples | filtered=%s | hits_ks=%s",
        test_triples.size(0),
        filtered,
        hits_ks,
    )

    metrics = evaluate_link_prediction(
        model=model,
        topo=None,
        eval_triples=test_triples,
        all_triples=all_triples,
        device=device,
        V=V_fused,
        batch_size_entities=batch_size_entities,
        filtered=filtered,
        hits_ks=hits_ks,
    )

    logger.info(
        "test metrics: " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    )

    print("Test set metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    test_config = {
        "testing": {"checkpoint_path": str(ckpt_path)},
        "data": {"prepared_dir": cfg.get("data", {}).get("prepared_dir")},
        "evaluation": {
            "filtered": filtered,
            "batch_size_entities": batch_size_entities,
            "hits_ks": hits_ks,
        },
        "runtime": {"device": str(device), "seed": seed},
    }

    with open(out_dir / "test_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(test_config, f, sort_keys=False)

    logger.info("saved test results to %s", out_dir)


if __name__ == "__main__":
    main()
