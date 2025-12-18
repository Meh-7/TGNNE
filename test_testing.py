"""
test_testing.py

Smoke test for:
  - data_prep.py
  - main_flex.py
  - testing.py

What it does:
  - builds a tiny KG with a few entities and one relation
  - writes toy train/valid/test txt files
  - runs data_prep.py to create a prepared dataset
  - runs main_flex.py for 1 epoch on CPU
  - locates the produced checkpoint
  - runs testing.py using that checkpoint
  - passes if no exception is raised

Run from project root:
    python test_testing.py
"""

import sys
import textwrap
import tempfile
from pathlib import Path

from data_prep import main as mvte_data_prep_main
from main_flex import main as mvte_train_main
from testing import main as mvte_test_main


# ---------------------------------------------------------------------
# Toy data
# ---------------------------------------------------------------------

def make_toy_data(root: Path) -> tuple[Path, Path, Path]:
    data_dir = root / "raw_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    triples_train = [
        ("e0", "r0", "e1"),
        ("e1", "r0", "e2"),
        ("e2", "r0", "e3"),
        ("e3", "r0", "e0"),
    ]
    triples_valid = [
        ("e0", "r0", "e2"),
        ("e1", "r0", "e3"),
    ]
    triples_test = [
        ("e2", "r0", "e0"),
        ("e3", "r0", "e1"),
    ]

    def write(path: Path, triples):
        with path.open("w", encoding="utf-8") as f:
            for h, r, t in triples:
                f.write(f"{h}\t{r}\t{t}\n")

    train = data_dir / "train.txt"
    valid = data_dir / "valid.txt"
    test = data_dir / "test.txt"

    write(train, triples_train)
    write(valid, triples_valid)
    write(test, triples_test)

    return train, valid, test


# ---------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------

def make_data_prep_config(
    root: Path,
    train: Path,
    valid: Path,
    test: Path,
) -> Path:
    cfg = f"""
    data:
      source: files
      train_path: "{train.as_posix()}"
      valid_path: "{valid.as_posix()}"
      test_path: "{test.as_posix()}"
      prepared_dir: "{(root / 'prepared').as_posix()}"

    topology:
      max_triangles_per_entity: 10
      max_tetras_per_entity: 10
    """
    path = root / "config_prep.yaml"
    path.write_text(textwrap.dedent(cfg))
    return path


def make_train_config(root: Path, prepared_dir: Path) -> Path:
    cfg = f"""
    data:
      prepared_dir: "{prepared_dir.as_posix()}"

    model:
      embedding_dim: 16
      tri_hidden_dim: 16
      tet_hidden_dim: 16
      dropout: 0.0
      gamma: 12.0
      base_scorer: transe
      fusion_mode: learned

    training:
      device: cpu
      lr: 0.001
      num_epochs: 1
      batch_size: 4
      num_negatives: 2
      adversarial_temperature: 1.0
      negative_mode: both
      log_interval: 10

    evaluation:
      batch_size_entities: 4
      filtered: true
      hits_ks: [1, 3, 10]

    output:
      save_dir: "{(root / 'runs').as_posix()}"
    """
    path = root / "config_train.yaml"
    path.write_text(textwrap.dedent(cfg))
    return path


def make_test_config(root: Path, prepared_dir: Path, ckpt: Path) -> Path:
    cfg = f"""
    data:
      prepared_dir: "{prepared_dir.as_posix()}"

    testing:
      checkpoint_path: "{ckpt.as_posix()}"

    training:
      device: cpu

    evaluation:
      batch_size_entities: 4
      filtered: true
      hits_ks: [1, 3, 10]
    """
    path = root / "config_test.yaml"
    path.write_text(textwrap.dedent(cfg))
    return path


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def run_with_argv(main_fn, argv):
    old = sys.argv
    try:
        sys.argv = argv
        main_fn()
    finally:
        sys.argv = old


# ---------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------

def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mvte_smoke_") as tmp:
        root = Path(tmp)

        # 1) raw data
        train, valid, test = make_toy_data(root)

        # 2) data prep
        prep_cfg = make_data_prep_config(root, train, valid, test)
        run_with_argv(mvte_data_prep_main, ["data_prep.py", "--config", str(prep_cfg)])

        prepared_dir = Path("data") / "train"
        assert prepared_dir.exists(), f"prepared dataset not found: {prepared_dir}"

        # 3) training
        train_cfg = make_train_config(root, prepared_dir)
        run_with_argv(mvte_train_main, ["main_flex.py", "--config", str(train_cfg)])

        # locate checkpoint
        run_root = root / "runs"
        assert run_root.exists(), "runs directory not found"

        # runs/<run_name>/
        run_name_dirs = sorted(d for d in run_root.iterdir() if d.is_dir())
        assert run_name_dirs, "no run-name directories created"

        # runs/<run_name>/<timestamp>/
        timestamp_dirs = sorted(d for d in run_name_dirs[-1].iterdir() if d.is_dir())
        assert timestamp_dirs, "no timestamped run directories created"

        ckpt = timestamp_dirs[-1] / "mvte_model.pt"
        assert ckpt.exists(), f"checkpoint not found: {ckpt}"

        # 4) testing
        test_cfg = make_test_config(root, prepared_dir, ckpt)
        run_with_argv(mvte_test_main, ["testing.py", "--config", str(test_cfg)])

        print("Smoke test for updated testing.py PASSED.")


if __name__ == "__main__":
    main()
