"""
test_testing.py

Smoke test for mvte/main.py + mvte/testing.py.

What it does:
  - builds a tiny KG with a few entities and one relation
  - writes toy train/valid/test txt files under a temporary directory
  - writes a config_smoke.yaml pointing at those files and an output dir
  - invokes main.main() once on CPU with num_epochs=1 (very short training)
  - checks that:
      * fused_entity_embeddings.pt exists
      * mvte_model.pt exists
      * encoded_splits/{train,valid,test}_ids.pt exist
  - invokes testing.main() with the same config
  - passes if no exception is raised

Run from project root:
    python test_testing.py
"""

import sys
import textwrap
import tempfile
from pathlib import Path

from main import main as mvte_train_main
from testing import main as mvte_test_main


def make_toy_data(root: Path) -> tuple[Path, Path, Path]:
    """Create tiny train/valid/test txt files with a small KG."""
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # simple KG over entities e0..e3 with a single relation r0
    triples_train = [
        ("e0", "r0", "e1"),
        ("e1", "r0", "e2"),
        ("e2", "r0", "e3"),
        ("e3", "r0", "e0"),
    ]
    # use subsets for valid / test just to differ slightly
    triples_valid = [
        ("e0", "r0", "e2"),
        ("e1", "r0", "e3"),
    ]
    triples_test = [
        ("e2", "r0", "e0"),
        ("e3", "r0", "e1"),
    ]

    def write_triples(path: Path, triples: list[tuple[str, str, str]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for h, r, t in triples:
                f.write(f"{h}\t{r}\t{t}\n")

    train_path = data_dir / "train.txt"
    valid_path = data_dir / "valid.txt"
    test_path = data_dir / "test.txt"

    write_triples(train_path, triples_train)
    write_triples(valid_path, triples_valid)
    write_triples(test_path, triples_test)

    return train_path, valid_path, test_path


def make_smoke_config(
    root: Path,
    train_path: Path,
    valid_path: Path,
    test_path: Path,
) -> Path:
    """Write a minimal config_smoke.yaml pointing to toy data + output."""
    cfg_path = root / "config_smoke.yaml"
    out_dir = root / "out"

    cfg_text = f"""
    data:
      train_path: "{train_path.as_posix()}"
      valid_path: "{valid_path.as_posix()}"
      test_path: "{test_path.as_posix()}"

    model:
      embedding_dim: 16
      tri_hidden_dim: 16
      tet_hidden_dim: 16
      dropout: 0.0
      gamma: 12.0

    topology:
      max_triangles_per_entity: 10
      max_tetras_per_entity: 10

    training:
      seed: 123
      device: "cpu"
      lr: 0.001
      num_epochs: 1
      batch_size: 4
      num_negatives: 2
      adversarial_temperature: 1.0
      negative_mode: "both"
      log_interval: 10

    evaluation:
      batch_size_entities: 4
      filtered: true
      hits_ks: [1, 3, 10]
      eval_every: 1

    logging:
      level: "INFO"
      log_dir: null

    output:
      save_dir: "{out_dir.as_posix()}"
    """

    cfg_path.write_text(textwrap.dedent(cfg_text), encoding="utf-8")
    return cfg_path


def run_train(config_path: Path) -> None:
    """Run main.main() with the given config path."""
    argv_backup = sys.argv.copy()
    try:
        sys.argv = ["main.py", "--config", str(config_path)]
        mvte_train_main()
    finally:
        sys.argv = argv_backup


def run_test(config_path: Path) -> None:
    """Run testing.main() with the given config path."""
    argv_backup = sys.argv.copy()
    try:
        sys.argv = ["testing.py", "--config", str(config_path)]
        mvte_test_main()
    finally:
        sys.argv = argv_backup


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mvte_smoke_") as tmpdir:
        root = Path(tmpdir)

        # 1) create toy data
        train_path, valid_path, test_path = make_toy_data(root)

        # 2) create smoke config
        cfg_path = make_smoke_config(root, train_path, valid_path, test_path)

        # 3) run training via main.py
        run_train(cfg_path)

        # 4) check that artifacts exist
        out_dir = root / "out"
        ckpt_path = out_dir / "mvte_model.pt"
        fused_path = out_dir / "fused_entity_embeddings.pt"
        encoded_dir = out_dir / "encoded_splits"
        train_ids = encoded_dir / "train_ids.pt"
        valid_ids = encoded_dir / "valid_ids.pt"
        test_ids = encoded_dir / "test_ids.pt"

        assert ckpt_path.exists(), f"checkpoint not found: {ckpt_path}"
        assert fused_path.exists(), f"fused embeddings not found: {fused_path}"
        assert encoded_dir.exists(), f"encoded_splits dir not found: {encoded_dir}"
        assert train_ids.exists(), f"train_ids.pt not found: {train_ids}"
        assert valid_ids.exists(), f"valid_ids.pt not found: {valid_ids}"
        assert test_ids.exists(), f"test_ids.pt not found: {test_ids}"

        # 5) run testing.py evaluation
        run_test(cfg_path)

        print("Smoke test for testing.py PASSED.")


if __name__ == "__main__":
    main()
