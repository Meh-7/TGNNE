# mvte/testing.py

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from evaluation import evaluate_link_prediction
from model import MVTEModel
from utils import (
    setup_logging,
    set_random_seed,
    get_device_from_config,
    get_run_name_from_config
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for test-time evaluation."""
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
    """Load YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main() -> None:
    """Entry point: load checkpoint, fused embeddings and encoded splits, then
    run link prediction evaluation on the test split."""
    args = parse_args()
    cfg = load_config(args.config)

    # logging & seed
    log_cfg = cfg.get("logging", {})
    log_level = log_cfg.get("level", "INFO")
    base_log_dir = log_cfg.get("log_dir", None)
    run_name = get_run_name_from_config(cfg)
    log_dir = None
    if base_log_dir is not None:
        # e.g. logs/FB15k237_test
        log_dir = str(Path(base_log_dir) / f"{run_name}_test")

    setup_logging(log_level=log_level, log_dir=log_dir)
    logger.info("loaded config from %s", args.config)

    # seeding should be useless in this eval script, but do it anyway for consistency
    seed = 123
    set_random_seed(seed)
    logger.info("random seed set to %d", seed)

    # device 
    train_cfg = cfg.get("training", {})
    device_str = train_cfg.get("device", "auto")
    device = get_device_from_config(device_str)
    logger.info("using device: %s", device)

    # locate output directory & artifacts 
    output_cfg = cfg.get("output", {})
    base_save_dir = output_cfg.get("save_dir", None)
    if base_save_dir is None:
        raise ValueError(
            "output.save_dir is not set in the config; cannot locate checkpoint "
            "and encoded splits for evaluation."
        )
    run_name = get_run_name_from_config(cfg)
    out_dir = Path(base_save_dir) / run_name

    if not out_dir.exists():
        raise FileNotFoundError(f"output directory not found: {out_dir}")

    ckpt_path = out_dir / "mvte_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint file not found: {ckpt_path}")

    fused_path = out_dir / "fused_entity_embeddings.pt"
    if not fused_path.exists():
        raise FileNotFoundError(
            f"fused entity embedding file not found: {fused_path}. "
            "Make sure training finished and saved embeddings."
        )

    encoded_dir = out_dir / "encoded_splits"
    if not encoded_dir.exists():
        raise FileNotFoundError(
            f"encoded_splits directory not found: {encoded_dir}. "
            "Make sure main.py was run with output.save_dir set."
        )

    train_ids_path = encoded_dir / "train_ids.pt"
    test_ids_path = encoded_dir / "test_ids.pt"
    valid_ids_path = encoded_dir / "valid_ids.pt"

    if not train_ids_path.exists():
        raise FileNotFoundError(f"train_ids.pt not found in {encoded_dir}")
    if not test_ids_path.exists():
        raise FileNotFoundError(
            f"test_ids.pt not found in {encoded_dir} – did you train with a test split?"
        )

    # load encoded splits
    logger.info("loading encoded splits from %s", encoded_dir)
    train_triples = torch.load(train_ids_path, map_location="cpu")
    test_triples = torch.load(test_ids_path, map_location="cpu")

    valid_triples = None
    if valid_ids_path.exists():
        valid_triples = torch.load(valid_ids_path, map_location="cpu")
        logger.info("loaded validation encoded triples as well")

    # Build union of all triples for filtered evaluation (on CPU).
    all_triples = train_triples
    if valid_triples is not None:
        all_triples = torch.cat([all_triples, valid_triples], dim=0)
    if test_triples is not None:
        all_triples = torch.cat([all_triples, test_triples], dim=0)

    # load checkpoint & reconstruct model 
    logger.info("loading checkpoint from %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    entity2id = checkpoint["entity2id"]
    relation2id = checkpoint["relation2id"]
    ckpt_cfg = checkpoint.get("config", cfg)  # fall back to current cfg if needed
    model_cfg = ckpt_cfg.get("model", {})

    num_entities = len(entity2id)
    num_relations = len(relation2id)

    embedding_dim = model_cfg.get("embedding_dim", 200)
    tri_hidden_dim = model_cfg.get("tri_hidden_dim", embedding_dim)
    tet_hidden_dim = model_cfg.get("tet_hidden_dim", embedding_dim)
    dropout = model_cfg.get("dropout", 0.0)
    gamma = model_cfg.get("gamma", 12.0)

    logger.info(
        "reconstructing MVTEModel: num_entities=%d | num_relations=%d | "
        "embedding_dim=%d | tri_hidden_dim=%d | tet_hidden_dim=%d | "
        "dropout=%.3f | gamma=%.3f",
        num_entities,
        num_relations,
        embedding_dim,
        tri_hidden_dim,
        tet_hidden_dim,
        dropout,
        gamma,
    )

    model = MVTEModel(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        tri_hidden_dim=tri_hidden_dim,
        tet_hidden_dim=tet_hidden_dim,
        dropout=dropout,
        gamma=gamma,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # load fused entity embeddings V 
    logger.info("loading fused entity embeddings from %s", fused_path)
    fused_data = torch.load(fused_path, map_location="cpu")
    V_fused = fused_data["V_fused"]  # [num_entities, embedding_dim]

    if V_fused.size(0) != num_entities:
        raise ValueError(
            f"V_fused has {V_fused.size(0)} rows but checkpoint has {num_entities} "
            "entities – mismatch between embeddings and id mapping."
        )

    # evaluation config (support both 'evaluation' and the typo key) 
    eval_cfg = cfg.get("evaluation", cfg.get("evaluatgion", {}))
    batch_size_entities = eval_cfg.get("batch_size_entities", 1024)
    filtered = eval_cfg.get("filtered", True)
    hits_ks = eval_cfg.get("hits_ks", [1, 3, 10])

    logger.info(
        "starting evaluation on test set: %d triples | filtered=%s | hits_ks=%s",
        test_triples.size(0),
        filtered,
        hits_ks,
    )

    # run evaluation on the test split
    metrics = evaluate_link_prediction(
        model=model,
        topo=None,  # we provide V_fused directly, so topology is not needed here
        eval_triples=test_triples,
        all_triples=all_triples,
        device=device,
        V=V_fused,
        batch_size_entities=batch_size_entities,
        filtered=filtered,
        hits_ks=hits_ks,
    )

    # Log and print metrics
    logger.info(
        "test metrics: " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    )

    print("Test set metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
