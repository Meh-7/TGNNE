# mvte/utils.py

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import torch


logger = logging.getLogger(__name__)


# seeding

def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed) # python's built-in random module
    np.random.seed(seed) # numpy random seed
    torch.manual_seed(seed) # torch cpu random seed
    if torch.cuda.is_available(): # torch gpu random seed
        torch.cuda.manual_seed_all(seed)


# logging 

def setup_logging(log_level: str = "INFO", log_dir: Optional[str] = None) -> None:
    """Configure root logger with optional file logging."""
    level = getattr(logging, log_level.upper(), logging.INFO) # get log level from string, default to INFO

    handlers: List[logging.Handler] = [logging.StreamHandler()] # console handler: print logs to console (stdout)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True) # create log directory if it doesn't exist
        log_file = Path(log_dir) / "train.log"
        file_handler = logging.FileHandler(log_file) # create a file logging handler that writes logs to that file.
        handlers.append(file_handler) # add file handler to handlers list
    # now logs go to both: console and file (if log_dir is set)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


# device

def get_device_from_config(device_str: str) -> torch.device:
    """Parse device config ('cpu', 'cuda', 'cuda:0', or 'auto')."""
    device_str = device_str.lower()
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


# data loading / id mapping 

TripleStr = Tuple[str, str, str]


def load_triples_from_file(
    path: Path,
    sep: str = "\t",
) -> List[TripleStr]:
    """Load triples (h, r, t) from a text file.

    Each line: head<sep>relation<sep>tail
    Entities and relations are kept as strings at this stage.
    """
    triples: List[TripleStr] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(sep)
            if len(parts) != 3:
                raise ValueError(f"Invalid triple line (expected 3 columns): {line}")
            h, r, t = parts
            triples.append((h, r, t))
    return triples


def build_id_mappings(
    train_triples: Sequence[TripleStr],
    valid_triples: Optional[Sequence[TripleStr]] = None,
    test_triples: Optional[Sequence[TripleStr]] = None,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build entity and relation id mappings from all splits.

    Ensures consistent ids across train/valid/test.
    """
    entities: Dict[str, int] = {}
    relations: Dict[str, int] = {}

    def add_triples(triples: Optional[Sequence[TripleStr]]) -> None:
        if triples is None:
            return
        for h, r, t in triples:
            if h not in entities:
                entities[h] = len(entities)
            if t not in entities:
                entities[t] = len(entities)
            if r not in relations:
                relations[r] = len(relations)

    add_triples(train_triples)
    add_triples(valid_triples)
    add_triples(test_triples)

    logger.info(
        "Built id mappings: %d entities, %d relations",
        len(entities),
        len(relations),
    )
    return entities, relations


def encode_triples(
    triples: Sequence[TripleStr],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
) -> torch.Tensor: # (actually returns LongTensor)
    """Encode string triples to integer index tensor [N, 3]."""
    encoded = []
    for h, r, t in triples:
        try:
            h_id = entity2id[h]
            t_id = entity2id[t]
            r_id = relation2id[r]
        except KeyError as e:
            raise KeyError(f"Unknown symbol in triples: {e}") from e
        encoded.append((h_id, r_id, t_id))

    if not encoded:
        return torch.empty((0, 3), dtype=torch.long)

    return torch.tensor(encoded, dtype=torch.long)

def get_run_name_from_config(cfg: Dict[str, Any]) -> str:
    """Derive a run/dataset name from the config for subdirectory naming."""
    data_cfg = cfg.get("data", {})
    source = data_cfg.get("source", "files")

    if source == "pykeen":
        # use the dataset name directly, e.g. "FB15k237"
        return str(data_cfg.get("dataset", "dataset"))
    else:
        # use train file stem, e.g. "train_fb15k237"
        train_path = Path(data_cfg["train_path"])
        return train_path.stem
