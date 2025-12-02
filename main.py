# mvte/main.py

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

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
    """entry point: load config, data, topology, model, and run training."""
    args = parse_args()
    cfg = load_config(args.config)

    # logging 
    log_level = cfg.get("logging", {}).get("level", "INFO")
    log_dir = cfg.get("logging", {}).get("log_dir", None)
    setup_logging(log_level=log_level, log_dir=log_dir)

    logger.info("loaded config from %s", args.config)

    # seeding 
    seed = cfg.get("training", {}).get("seed", 42)
    set_random_seed(seed)
    logger.info("random seed set to %d", seed)

    # device 
    device = get_device_from_config(cfg.get("training", {}).get("device", "auto"))
    logger.info("using device: %s", device)

    # paths
    data_cfg = cfg.get("data", {})
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

    output_dir = cfg.get("output", {}).get("save_dir", None)

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

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
        metrics = evaluate_link_prediction(
            model=model,
            topo=topo,
            eval_triples=valid_triples,
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

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
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

    logger.info("done")


if __name__ == "__main__":
    main()
