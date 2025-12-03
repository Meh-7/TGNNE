"""
Smoke tests for data_prep.py.

Usage:
    python smoke_test_data_prep.py
"""
import sys

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import torch
import yaml

import data_prep  # assumes this script lives in the same pkg as data_prep.py


Triple = Tuple[str, str, str]


def _write_triples(path: Path, triples: List[Triple]) -> None:
    lines = ["\t".join(triple) for triple in triples]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _check_dataset_dir(dataset_dir: Path) -> None:
    """Assert that a prepared dataset directory looks correct."""
    assert dataset_dir.exists(), f"{dataset_dir} does not exist"

    # Check split tensors
    train_path = dataset_dir / "train_ids.pt"
    assert train_path.exists(), "train_ids.pt missing"
    train_ids = torch.load(train_path)
    assert train_ids.ndim == 2 and train_ids.shape[1] == 3, "train_ids shape invalid"
    assert train_ids.dtype == torch.long, "train_ids dtype must be long"

    valid_path = dataset_dir / "valid_ids.pt"
    test_path = dataset_dir / "test_ids.pt"
    assert valid_path.exists(), "valid_ids.pt missing"
    assert test_path.exists(), "test_ids.pt missing"

    valid_ids = torch.load(valid_path)
    test_ids = torch.load(test_path)

    for name, tensor in [("valid", valid_ids), ("test", test_ids)]:
        assert tensor.ndim == 2 and tensor.shape[1] == 3, f"{name}_ids shape invalid"
        assert tensor.dtype == torch.long, f"{name}_ids dtype must be long"

    # Check mappings
    ent_path = dataset_dir / "entity2id.pt"
    rel_path = dataset_dir / "relation2id.pt"
    assert ent_path.exists(), "entity2id.pt missing"
    assert rel_path.exists(), "relation2id.pt missing"

    entity2id = torch.load(ent_path)
    relation2id = torch.load(rel_path)
    assert isinstance(entity2id, dict), "entity2id must be a dict"
    assert isinstance(relation2id, dict), "relation2id must be a dict"
    assert len(entity2id) > 0, "entity2id is empty"
    assert len(relation2id) > 0, "relation2id is empty"

    # JSON inverses
    id2entity_path = dataset_dir / "id2entity.json"
    id2relation_path = dataset_dir / "id2relation.json"
    assert id2entity_path.exists(), "id2entity.json missing"
    assert id2relation_path.exists(), "id2relation.json missing"

    id2entity = json.loads(id2entity_path.read_text(encoding="utf-8"))
    id2relation = json.loads(id2relation_path.read_text(encoding="utf-8"))
    # Quick consistency checks
    assert len(id2entity) == len(entity2id), "id2entity size mismatch"
    assert len(id2relation) == len(relation2id), "id2relation size mismatch"

    # meta.yaml
    meta_path = dataset_dir / "meta.yaml"
    assert meta_path.exists(), "meta.yaml missing"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    for key in ["name", "source", "num_entities", "num_relations", "num_triples"]:
        assert key in meta, f"meta.yaml missing key '{key}'"

    print(f"[OK] Checked dataset dir: {dataset_dir}")


def test_prepare_from_files_direct() -> None:
    """Test the prepare_from_files() API directly."""
    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_root = Path(tmp_root)
        # Raw triple files
        raw_dir = tmp_root / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        train_triples: List[Triple] = [
            ("e1", "r1", "e2"),
            ("e2", "r1", "e3"),
            ("e3", "r2", "e1"),
        ]
        valid_triples: List[Triple] = [
            ("e1", "r2", "e3"),
        ]
        test_triples: List[Triple] = [
            ("e2", "r2", "e1"),
        ]

        train_path = raw_dir / "train.txt"
        valid_path = raw_dir / "valid.txt"
        test_path = raw_dir / "test.txt"

        _write_triples(train_path, train_triples)
        _write_triples(valid_path, valid_triples)
        _write_triples(test_path, test_triples)

        dataset_dir = tmp_root / "toy_ds_direct"

        # Call the function entrypoint
        data_prep.prepare_from_files(
            train_path=train_path,
            valid_path=valid_path,
            test_path=test_path,
            dataset_dir=dataset_dir,
        )

        _check_dataset_dir(dataset_dir)
        print("[OK] test_prepare_from_files_direct passed")


def test_cli_files_mode() -> None:
    """Test the CLI entrypoint: python data_prep.py --source files ..."""
    import inspect

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_root = Path(tmp_root)
        raw_dir = tmp_root / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        train_triples: List[Triple] = [
            ("x1", "rA", "x2"),
            ("x2", "rB", "x1"),
        ]
        valid_triples: List[Triple] = [
            ("x1", "rA", "x1"),
        ]
        test_triples: List[Triple] = [
            ("x2", "rB", "x2"),
        ]

        train_path = raw_dir / "train_cli.txt"
        valid_path = raw_dir / "valid_cli.txt"
        test_path = raw_dir / "test_cli.txt"

        _write_triples(train_path, train_triples)
        _write_triples(valid_path, valid_triples)
        _write_triples(test_path, test_triples)

        output_root = tmp_root / "out"
        output_root.mkdir(parents=True, exist_ok=True)

        dataset_name = "toy_cli"
        dataset_dir = output_root / dataset_name

        # Locate data_prep.py on disk
        data_prep_path = Path(inspect.getfile(data_prep))

        # Run the script as a subprocess to exercise argparse + main()
        cmd = [
            sys.executable,  # use the same Python that runs this test
            str(data_prep_path),
            "--source",
            "files",
            "--train-path",
            str(train_path),
            "--valid-path",
            str(valid_path),
            "--test-path",
            str(test_path),
            "--output-root",
            str(output_root),
            "--name",
            dataset_name,
        ]

        print("[INFO] Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        _check_dataset_dir(dataset_dir)
        print("[OK] test_cli_files_mode passed")


if __name__ == "__main__":
    test_prepare_from_files_direct()
    test_cli_files_mode()
    print("All smoke tests passed.")
