# main.py
# CLI: preprocess / train / eval

from __future__ import annotations
import argparse
import os
import json

import numpy as np
import torch

from utils import (
    set_seed,
    log,
    ensure_dir,
    get_device,
    save_id_maps,
    load_id_maps,
)
from mvte.data_topology_networkx import (
    load_triples,
    build_id_maps,
    encode_triples,
    split_triples,
    build_base_graph,
    find_triangles,
    find_tetrahedra,
    build_simplex_tree,
    make_views_from_simplex_tree,
    save_views,
    load_views,
    views_summary_str,
)
from model import MVTEModel
from training import BernoulliNegativeSampler, train_epoch, evaluate_tail


# -----------------------------
# Preprocess
# -----------------------------

def cmd_preprocess(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    log(f"Loading triples from {args.triples}")
    triples = load_triples(args.triples, sep=args.sep)
    log(f"Loaded {len(triples)} triples")

    log("Building ID maps")
    maps = build_id_maps(triples)
    num_entities = len(maps.id2ent)
    num_relations = len(maps.id2rel)
    log(f"Entities={num_entities} Relations={num_relations}")

    log("Encoding triples to IDs")
    id_triples = encode_triples(triples, maps)

    log("Splitting triples")
    ratios = (args.train_ratio, args.valid_ratio, args.test_ratio)
    train, valid, test = split_triples(id_triples, ratios=ratios, seed=args.seed)
    log(f"Split sizes: train={len(train)} valid={len(valid)} test={len(test)}")

    # Build topology on full graph (could also restrict to train)
    log("Building base graph")
    G = build_base_graph(id_triples, num_entities=num_entities)
    log(f"Graph nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    log("Finding triangles")
    triangles = find_triangles(G, cap_per_node=args.triangle_cap, seed=args.seed)
    log(f"Triangles found={len(triangles)}")

    log("Finding tetrahedra")
    tetras = find_tetrahedra(G, cap_per_node=args.tetra_cap, seed=args.seed)
    log(f"Tetrahedra found={len(tetras)}")

    log("Building simplex tree")
    edges = [(u, v) for u, v in G.edges()]
    st = build_simplex_tree(edges, triangles, tetras)

    log("Building views from simplex tree")
    views = make_views_from_simplex_tree(st, num_entities=num_entities)
    log("Views: " + views_summary_str(views))

    # Save artifacts
    splits_path = os.path.join(args.out_dir, "splits.npz")
    views_path = os.path.join(args.out_dir, "views.npz")
    maps_path = os.path.join(args.out_dir, "id_maps.json")
    meta_path = os.path.join(args.out_dir, "meta.json")

    np.savez(splits_path, train=train, valid=valid, test=test)
    save_views(views_path, views)
    save_id_maps(maps_path, maps)

    meta = {
        "num_entities": num_entities,
        "num_relations": num_relations,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log("Preprocess done")


# -----------------------------
# Train
# -----------------------------

def cmd_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    log(f"Using device: {device}")

    splits_path = os.path.join(args.data_dir, "splits.npz")
    views_path = os.path.join(args.data_dir, "views.npz")
    maps_path = os.path.join(args.data_dir, "id_maps.json")
    meta_path = os.path.join(args.data_dir, "meta.json")

    if not os.path.exists(splits_path):
        raise FileNotFoundError(splits_path)
    if not os.path.exists(views_path):
        raise FileNotFoundError(views_path)
    if not os.path.exists(maps_path):
        raise FileNotFoundError(maps_path)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)

    log("Loading splits")
    splits = np.load(splits_path)
    train = splits["train"]
    valid = splits["valid"]
    test = splits["test"]

    log("Loading views")
    views = load_views(views_path)
    log("Loading ID maps and meta")
    maps = load_id_maps(maps_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_entities = meta["num_entities"]
    num_relations = meta["num_relations"]

    log(f"num_entities={num_entities} num_relations={num_relations}")
    log("Building model")

    model = MVTEModel(
        num_entities=num_entities,
        num_relations=num_relations,
        dim=args.dim,
        num_triangles=views.num_triangles,
        num_tetras=views.num_tetras,
        simplex_init=args.simplex_init,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sampler = BernoulliNegativeSampler(train, num_entities=num_entities, seed=args.seed)

    # Save training config
    train_meta = {
        "dim": args.dim,
        "simplex_init": args.simplex_init,
    }
    with open(os.path.join(args.data_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            train_triples=train,
            views=views,
            sampler=sampler,
            batch_size=args.batch_size,
            neg_k=args.neg_k,
            adv_alpha=args.adv_alpha,
            device=device,
            epoch=epoch,
            log_every=args.log_every,
        )
        with torch.no_grad():
            alpha = model.fusion.alpha.detach().cpu()
            w = torch.softmax(alpha, dim=0)
        log(f"Epoch {epoch} avg_loss={avg_loss:.4f} alpha={alpha.tolist()} w={w.tolist()}")

    ckpt_path = os.path.join(args.data_dir, "model.pt")
    ckpt = {
        "state_dict": model.state_dict(),
        "meta": {
            "num_entities": num_entities,
            "num_relations": num_relations,
            "dim": args.dim,
            "num_triangles": views.num_triangles,
            "num_tetras": views.num_tetras,
            "simplex_init": args.simplex_init,
        },
    }
    torch.save(ckpt, ckpt_path)
    log(f"Saved checkpoint to {ckpt_path}")


# -----------------------------
# Eval
# -----------------------------

def cmd_eval(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    log(f"Using device: {device}")

    splits_path = os.path.join(args.data_dir, "splits.npz")
    views_path = os.path.join(args.data_dir, "views.npz")
    ckpt_path = os.path.join(args.data_dir, args.checkpoint)

    if not os.path.exists(splits_path):
        raise FileNotFoundError(splits_path)
    if not os.path.exists(views_path):
        raise FileNotFoundError(views_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    log("Loading splits")
    splits = np.load(splits_path)
    train = splits["train"]
    valid = splits["valid"]
    test = splits["test"]
    all_triples = np.concatenate([train, valid, test], axis=0)

    log("Loading views")
    views = load_views(views_path)

    log("Loading checkpoint")
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["meta"]

    num_entities = meta["num_entities"]
    num_relations = meta["num_relations"]
    dim = meta["dim"]
    simplex_init = meta["simplex_init"]
    num_triangles = meta["num_triangles"]
    num_tetras = meta["num_tetras"]

    model = MVTEModel(
        num_entities=num_entities,
        num_relations=num_relations,
        dim=dim,
        num_triangles=num_triangles,
        num_tetras=num_tetras,
        simplex_init=simplex_init,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])

    log("Running tail-only filtered evaluation on test set")
    metrics = evaluate_tail(
        model=model,
        views=views,
        test_triples=test,
        all_triples=all_triples,
        device=device,
    )
    for k, v in metrics.items():
        log(f"{k} = {v:.4f}")


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-View Topology-Aware KG Embedding")
    sub = parser.add_subparsers(dest="command", required=True)

    # preprocess
    p_pre = sub.add_parser("preprocess", help="triples→ids→graph→cliques→SimplexTree→views")
    p_pre.add_argument("--triples", type=str, required=True, help="Input triples file (h r t)")
    p_pre.add_argument("--sep", type=str, default="\t", help="Separator in triples file")
    p_pre.add_argument("--out-dir", type=str, required=True, help="Output directory")
    p_pre.add_argument("--train-ratio", type=float, default=0.8)
    p_pre.add_argument("--valid-ratio", type=float, default=0.1)
    p_pre.add_argument("--test-ratio", type=float, default=0.1)
    p_pre.add_argument("--triangle-cap", type=int, default=None, help="Max triangles per node (optional)")
    p_pre.add_argument("--tetra-cap", type=int, default=None, help="Max tetrahedra per node (optional)")
    p_pre.add_argument("--seed", type=int, default=0)

    # train
    p_train = sub.add_parser("train", help="Train fused model")
    p_train.add_argument("--data-dir", type=str, required=True, help="Directory with preprocess artifacts")
    p_train.add_argument("--dim", type=int, default=200)
    p_train.add_argument("--batch-size", type=int, default=1024)
    p_train.add_argument("--neg-k", type=int, default=64)
    p_train.add_argument("--adv-alpha", type=float, default=2.0)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--simplex-init", type=str, default="mean", choices=["mean", "embed"])
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p_train.add_argument("--log-every", type=int, default=100)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate checkpoint")
    p_eval.add_argument("--data-dir", type=str, required=True, help="Directory with artifacts and model")
    p_eval.add_argument("--checkpoint", type=str, default="model.pt", help="Checkpoint file name")
    p_eval.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
