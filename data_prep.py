# mvte/data_prep.py
"""
Standalone dataset preparation script for MVTE.

Goal
-----
Prepare reusable dataset directory under a given root, e.g.:

    data/FB15k-237/
        train_ids.pt
        valid_ids.pt
        test_ids.pt
        entity2id.pt
        relation2id.pt
        id2entity.json        (optional but convenient)
        id2relation.json
        meta.yaml

After running this script, `main_flex.py` and `testing.py` can simply load
these tensors/mappings instead of re-reading raw files or PyKEEN datasets.

Usage examples
--------------

1) Using an existing config.yaml (with the current [data] section):

    python data_prep.py --config config.yaml --output-root data

2) Local files mode without a config:

    python data_prep.py \
        --source files \
        --train-path path/to/train.txt \
        --valid-path path/to/valid.txt \
        --test-path path/to/test.txt  \
        --name FB15k-237 \
        --output-root data

3) PyKEEN dataset mode without a config:

    python data_prep.py \
        --source pykeen \
        --dataset FB15k237 \
        --name FB15k-237 \
        --output-root data
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml

from utils import (
    load_triples_from_file,
    build_id_mappings,
    encode_triples,
    get_run_name_from_config,
    setup_logging,
)

logger = logging.getLogger(__name__)


# helpers


def _ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def _save_mappings(
    out_dir: Path,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
) -> None:
    """Save entity/relation mappings and their inverses."""
    _ensure_dir(out_dir)

    # torch-serialised Python dicts (easy to load back in code)
    torch.save(entity2id, out_dir / "entity2id.pt")
    torch.save(relation2id, out_dir / "relation2id.pt")

    # convenient JSON inverses for inspection / notebooks
    id2entity = {int(idx): ent for ent, idx in entity2id.items()}
    id2relation = {int(idx): rel for rel, idx in relation2id.items()}

    with (out_dir / "id2entity.json").open("w", encoding="utf-8") as f:
        json.dump(id2entity, f, ensure_ascii=False, indent=2)
    with (out_dir / "id2relation.json").open("w", encoding="utf-8") as f:
        json.dump(id2relation, f, ensure_ascii=False, indent=2)

    logger.info(
        "Saved mappings: %d entities, %d relations",
        len(entity2id),
        len(relation2id),
    )


def _save_split_tensor(out_dir: Path, name: str, tensor: Optional[torch.Tensor]) -> None:
    """Save a split tensor if it is not None."""
    if tensor is None:
        return
    _ensure_dir(out_dir)
    path = out_dir / f"{name}_ids.pt"
    torch.save(tensor.cpu(), path)
    logger.info("Saved %s split to %s (shape=%s)", name, path, tuple(tensor.shape))


def _write_meta(
    out_dir: Path,
    *,
    dataset_name: str,
    source: str,
    num_entities: int,
    num_relations: int,
    num_triples: Dict[str, int],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a small meta.yaml with dataset summary."""
    meta = {
        "name": dataset_name,
        "source": source,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_entities": num_entities,
        "num_relations": num_relations,
        "num_triples": num_triples,
    }
    if extra:
        meta["extra"] = extra

    path = out_dir / "meta.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f)
    logger.info("Wrote metadata to %s", path)


# preparation backends



def prepare_from_files(
    * ,
    train_path: Path,
    valid_path: Optional[Path],
    test_path: Optional[Path],
    dataset_dir: Path,
) -> None:
    """Prepare dataset from raw local triple files (string entities/relations)."""
    logger.info("Preparing dataset from local files into %s", dataset_dir)
    _ensure_dir(dataset_dir)

    # 1) Load raw triples
    logger.info("Loading training triples from %s", train_path)
    train_triples_raw = load_triples_from_file(train_path)

    valid_triples_raw = None
    test_triples_raw = None
    if valid_path is not None:
        logger.info("Loading validation triples from %s", valid_path)
        valid_triples_raw = load_triples_from_file(valid_path)
    if test_path is not None:
        logger.info("Loading test triples from %s", test_path)
        test_triples_raw = load_triples_from_file(test_path)

    # 2) Build id mappings
    entity2id, relation2id = build_id_mappings(
        train_triples_raw,
        valid_triples_raw,
        test_triples_raw,
    )
    num_entities = len(entity2id)
    num_relations = len(relation2id)

    # 3) Encode splits
    train_ids = encode_triples(train_triples_raw, entity2id, relation2id)
    valid_ids = (
        encode_triples(valid_triples_raw, entity2id, relation2id)
        if valid_triples_raw is not None
        else None
    )
    test_ids = (
        encode_triples(test_triples_raw, entity2id, relation2id)
        if test_triples_raw is not None
        else None
    )

    # 4) Save splits and mappings
    _save_split_tensor(dataset_dir, "train", train_ids)
    _save_split_tensor(dataset_dir, "valid", valid_ids)
    _save_split_tensor(dataset_dir, "test", test_ids)
    _save_mappings(dataset_dir, entity2id, relation2id)

    num_triples = {
        "train": int(train_ids.shape[0]),
        "valid": int(valid_ids.shape[0]) if valid_ids is not None else 0,
        "test": int(test_ids.shape[0]) if test_ids is not None else 0,
    }

    extra = {
        "train_path": str(train_path),
        "valid_path": str(valid_path) if valid_path is not None else None,
        "test_path": str(test_path) if test_path is not None else None,
    }

    _write_meta(
        dataset_dir,
        dataset_name=dataset_dir.name,
        source="files",
        num_entities=num_entities,
        num_relations=num_relations,
        num_triples=num_triples,
        extra=extra,
    )


def prepare_from_pykeen(
    *,
    dataset_name: str,
    dataset_dir: Path,
) -> None:
    """Prepare dataset from a PyKEEN dataset (integer-mapped triples)."""
    logger.info("Preparing dataset from PyKEEN '%s' into %s", dataset_name, dataset_dir)
    _ensure_dir(dataset_dir)

    try:
        from pykeen.datasets import FB15k237, get_dataset
    except ImportError as e:
        raise ImportError(
            "PyKEEN is required for source='pykeen'. "
            "Install it with `pip install pykeen`."
        ) from e

    # Use FB15k237 class directly, others via get_dataset
    if dataset_name == "FB15k237":
        ds = FB15k237()
    else:
        ds = get_dataset(dataset=dataset_name)

    # mapped_triples are already integer-indexed tensors [N, 3]
    train_ids = ds.training.mapped_triples.long()
    valid_ids = (
        ds.validation.mapped_triples.long() if ds.validation is not None else None
    )
    test_ids = ds.testing.mapped_triples.long() if ds.testing is not None else None

    entity2id = dict(ds.entity_to_id)
    relation2id = dict(ds.relation_to_id)

    num_entities = len(entity2id)
    num_relations = len(relation2id)

    _save_split_tensor(dataset_dir, "train", train_ids)
    _save_split_tensor(dataset_dir, "valid", valid_ids)
    _save_split_tensor(dataset_dir, "test", test_ids)
    _save_mappings(dataset_dir, entity2id, relation2id)

    num_triples = {
        "train": int(train_ids.shape[0]),
        "valid": int(valid_ids.shape[0]) if valid_ids is not None else 0,
        "test": int(test_ids.shape[0]) if test_ids is not None else 0,
    }

    extra = {
        "pykeen_dataset": dataset_name,
    }

    _write_meta(
        dataset_dir,
        dataset_name=dataset_dir.name,
        source="pykeen",
        num_entities=num_entities,
        num_relations=num_relations,
        num_triples=num_triples,
        extra=extra,
    )


# argument / config handling


def _load_config(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _derive_dataset_name(
    *,
    args: argparse.Namespace,
    cfg: Optional[Dict[str, Any]],
    data_cfg: Optional[Dict[str, Any]],
) -> str:
    """Decide the dataset subdirectory name under output_root."""
    if args.name:
        return args.name

    # If config exists, reuse existing helper
    if cfg is not None and data_cfg is not None:
        try:
            return get_run_name_from_config(cfg)
        except Exception:
            pass  # fall through to heuristics

    # Fallback heuristics
    source = (data_cfg or {}).get("source", args.source)
    if source == "pykeen":
        if data_cfg is not None and "dataset" in data_cfg:
            return str(data_cfg["dataset"])
        if args.dataset:
            return args.dataset
        return "pykeen_dataset"
    else:
        # derive from train path stem
        train_path = None
        if data_cfg is not None and "train_path" in data_cfg:
            train_path = Path(data_cfg["train_path"])
        elif args.train_path:
            train_path = Path(args.train_path)

        if train_path is not None:
            return train_path.stem

        return "dataset"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare datasets into  tensor format.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config with a [data] section (like main_flex).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data",
        help="Root directory in which to create the prepared dataset folder.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the prepared dataset subdirectory (overrides config).",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["files", "pykeen"],
        default=None,
        help="Data source (overrides config.data.source if given).",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Train triples file (for source=files when no config is used).",
    )
    parser.add_argument(
        "--valid-path",
        type=str,
        default=None,
        help="Validation triples file (optional, source=files).",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Test triples file (optional, source=files).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="PyKEEN dataset name (for source=pykeen when no config is used).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )

    args = parser.parse_args()

    # basic logging (console only); reuse project's helper
    setup_logging(log_level=args.log_level, log_dir=None)

    cfg = _load_config(args.config)
    data_cfg: Optional[Dict[str, Any]] = None
    if cfg is not None:
        data_cfg = cfg.get("data", {})

    # Determine source
    source = args.source or (data_cfg.get("source") if data_cfg is not None else None)
    if source is None:
        raise ValueError(
            "Data source is not specified. "
            "Use --source {files,pykeen} or provide a config with data.source."
        )

    # Compute dataset name and target directory
    dataset_name = _derive_dataset_name(args=args, cfg=cfg, data_cfg=data_cfg)
    output_root = Path(args.output_root)
    dataset_dir = output_root / dataset_name

    logger.info("Output root         : %s", output_root)
    logger.info("Dataset name        : %s", dataset_name)
    logger.info("Prepared dataset dir: %s", dataset_dir)
    logger.info("Source              : %s", source)

    if source == "files":
        # derive file paths from config or CLI
        if data_cfg is not None and "train_path" in data_cfg:
            train_path = Path(data_cfg["train_path"])
            valid_path = (
                Path(data_cfg["valid_path"]) if data_cfg.get("valid_path") else None
            )
            test_path = (
                Path(data_cfg["test_path"]) if data_cfg.get("test_path") else None
            )
        else:
            if args.train_path is None:
                raise ValueError(
                    "For source='files', you must provide either "
                    "data.train_path in the config or --train-path on the CLI."
                )
            train_path = Path(args.train_path)
            valid_path = Path(args.valid_path) if args.valid_path is not None else None
            test_path = Path(args.test_path) if args.test_path is not None else None

        prepare_from_files(
            train_path=train_path,
            valid_path=valid_path,
            test_path=test_path,
            dataset_dir=dataset_dir,
        )

    elif source == "pykeen":
        # derive dataset name from config or CLI
        if data_cfg is not None and "dataset" in data_cfg:
            pykeen_name = str(data_cfg["dataset"])
        elif args.dataset is not None:
            pykeen_name = args.dataset
        else:
            raise ValueError(
                "For source='pykeen', you must provide either "
                "data.dataset in the config or --dataset on the CLI."
            )

        prepare_from_pykeen(
            dataset_name=pykeen_name,
            dataset_dir=dataset_dir,
        )

    logger.info("Done preparing dataset '%s' in %s", dataset_name, dataset_dir)


if __name__ == "__main__":
    main()
