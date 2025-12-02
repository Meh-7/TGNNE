"""
test_main.py

Minimal smoke test for mvte/main.py:
  - builds a tiny KG with 4 entities forming a 4-clique (so you get triangles + a tetra)
  - writes it to smoke_data/train.txt, smoke_data/valid.txt, smoke_data/test.txt
  - writes config_smoke.yaml pointing to those files
  - invokes main.main() once on CPU with num_epochs=1 (very short training)
  - checks that:
        - encoded_splits/{train,valid,test}_ids.pt are created
        - fused_entity_embeddings.pt is created
        - mvte_model.pt checkpoint is created

Run from project root:
    python test_main.py
"""

import sys
from pathlib import Path
import textwrap

from main import main as mvte_main  # your existing entry point


def make_toy_data(root: Path) -> tuple[Path, Path, Path]:
    """Create tiny train/valid/test txt files with a 4-clique over entities e0..e3."""
    data_dir = root / "smoke_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.txt"
    valid_path = data_dir / "valid.txt"
    test_path = data_dir / "test.txt"

    # Undirected 4-clique via 6 triples; all use the same relation r0
    triples = [
        ("e0", "r0", "e1"),
        ("e0", "r0", "e2"),
        ("e0", "r0", "e3"),
        ("e1", "r0", "e2"),
        ("e1", "r0", "e3"),
        ("e2", "r0", "e3"),
    ]

    # For the smoke test, we can reuse the same triples for all splits
    for path in (train_path, valid_path, test_path):
        with path.open("w", encoding="utf-8") as f:
            for h, r, t in triples:
                f.write(f"{h}\t{r}\t{t}\n")

    return train_path, valid_path, test_path


def make_toy_config(root: Path, train_path: Path, valid_path: Path, test_path: Path) -> Path:
    """Write a minimal config_smoke.yaml pointing to the toy train/valid/test files."""
    cfg_path = root / "config_smoke.yaml"
    log_dir = root / "smoke_logs"
    out_dir = root / "smoke_checkpoints"
    log_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    # IMPORTANT: lr is numeric, not quoted
    cfg_text = f"""
data:
  train_path: "{train_path.as_posix()}"
  valid_path: "{valid_path.as_posix()}"
  test_path: "{test_path.as_posix()}"

model:
  embedding_dim: 16
  tri_hidden_dim: 16
  tet_hidden_dim: 16
  dropout: 0.1
  gamma: 12.0

topology:
  max_triangles_per_entity: null
  max_tetras_per_entity: null

training:
  device: "cpu"
  seed: 123
  lr: 0.001
  num_epochs: 1
  batch_size: 4
  num_negatives: 2
  adversarial_temperature: 1.0
  negative_mode: "both"
  log_interval: 1

evaluation:
  batch_size_entities: 4    # we only have 4 entities in the toy KG
  filtered: true
  hits_ks: [1, 3, 4]
  eval_every: 1             # run validation every epoch (here: just once)

logging:
  level: "INFO"
  log_dir: "{log_dir.as_posix()}"

output:
  save_dir: "{out_dir.as_posix()}"
"""

    cfg_path.write_text(textwrap.dedent(cfg_text).lstrip(), encoding="utf-8")
    return cfg_path


def run_smoke_test() -> None:
    """Run main() once with a toy config + toy data and check outputs."""
    project_root = Path(__file__).resolve().parent
    train_path, valid_path, test_path = make_toy_data(project_root)
    cfg_path = make_toy_config(project_root, train_path, valid_path, test_path)

    print(f"[mvte smoke] using config: {cfg_path}")

    # Fake CLI args so main.parse_args() sees our config
    sys.argv = ["main.py", "--config", str(cfg_path)]
    mvte_main()

    # Check that encoded splits were saved
    out_dir = project_root / "smoke_checkpoints"
    encoded_dir = out_dir / "encoded_splits"
    train_ids = encoded_dir / "train_ids.pt"
    valid_ids = encoded_dir / "valid_ids.pt"
    test_ids = encoded_dir / "test_ids.pt"

    assert train_ids.exists(), f"missing encoded train ids at {train_ids}"
    assert valid_ids.exists(), f"missing encoded valid ids at {valid_ids}"
    assert test_ids.exists(), f"missing encoded test ids at {test_ids}"

    # Check that fused embeddings and model checkpoint were saved by main.py
    fused_path = out_dir / "fused_entity_embeddings.pt"
    model_path = out_dir / "mvte_model.pt"

    assert fused_path.exists(), f"missing fused embeddings at {fused_path}"
    assert model_path.exists(), f"missing model checkpoint at {model_path}"

    print(f"[mvte smoke] encoded_splits present in {encoded_dir} ✔")
    print(f"[mvte smoke] fused embeddings: {fused_path} ✔")
    print(f"[mvte smoke] model checkpoint: {model_path} ✔")
    print("[mvte smoke] finished without crashing ✔")


if __name__ == "__main__":
    run_smoke_test()
