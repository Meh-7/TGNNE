"""
test_main_flex.py

Smoke test for mvte/main_flex.py:

  1) FILES BRANCH
     - builds a tiny KG with 4 entities forming a 4-clique (so you get triangles + a tetra)
     - writes it to smoke_data/train.txt, smoke_data/valid.txt, smoke_data/test.txt
     - writes config_smoke_files.yaml with data.source: "files"
     - invokes main_flex.main() once on CPU with num_epochs=1
     - checks that:
           - encoded_splits/{train,valid,test}_ids.pt are created
           - fused_entity_embeddings.pt is created
           - mvte_model.pt checkpoint is created

  2) PYKEEN BRANCH (optional)
     - only runs if pykeen is importable
     - writes config_smoke_pykeen.yaml with data.source: "pykeen", dataset: "FB15k237"
     - sets num_epochs=0 to avoid heavy training
     - invokes main_flex.main() and checks that fused embeddings + checkpoint exist

Run from project root:
    python test_main_flex.py
"""

import sys
from pathlib import Path
import textwrap

from main_flex import main as mvte_main  # entry point under test


# ---------- helpers for FILES branch ----------

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

    # Reuse the same triples for all splits
    for path in (train_path, valid_path, test_path):
        with path.open("w", encoding="utf-8") as f:
            for h, r, t in triples:
                f.write(f"{h}\t{r}\t{t}\n")

    return train_path, valid_path, test_path


def make_toy_config_files(
    root: Path,
    train_path: Path,
    valid_path: Path,
    test_path: Path,
) -> Path:
    """Write a minimal config_smoke_files.yaml pointing to the toy files (source=files)."""
    cfg_path = root / "config_smoke_files.yaml"
    log_dir = root / "smoke_logs_files"
    out_dir = root / "smoke_checkpoints_files"
    log_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    cfg_text = f"""
data:
  source: "files"
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


def run_smoke_test_files() -> None:
    """Run main_flex() once with a toy 'files' config + toy data and check outputs."""
    project_root = Path(__file__).resolve().parent
    train_path, valid_path, test_path = make_toy_data(project_root)
    cfg_path = make_toy_config_files(project_root, train_path, valid_path, test_path)

    print(f"[mvte smoke/files] using config: {cfg_path}")

    # Fake CLI args so main_flex.parse_args() sees our config
    sys.argv = ["main_flex.py", "--config", str(cfg_path)]
    mvte_main()

    # Check that encoded splits were saved
    out_dir = project_root / "smoke_checkpoints_files"
    encoded_dir = out_dir / "encoded_splits"
    train_ids = encoded_dir / "train_ids.pt"
    valid_ids = encoded_dir / "valid_ids.pt"
    test_ids = encoded_dir / "test_ids.pt"

    assert train_ids.exists(), f"missing encoded train ids at {train_ids}"
    assert valid_ids.exists(), f"missing encoded valid ids at {valid_ids}"
    assert test_ids.exists(), f"missing encoded test ids at {test_ids}"

    # Check that fused embeddings and model checkpoint were saved
    fused_path = out_dir / "fused_entity_embeddings.pt"
    model_path = out_dir / "mvte_model.pt"

    assert fused_path.exists(), f"missing fused embeddings at {fused_path}"
    assert model_path.exists(), f"missing model checkpoint at {model_path}"

    print(f"[mvte smoke/files] encoded_splits present in {encoded_dir} ✔")
    print(f"[mvte smoke/files] fused embeddings: {fused_path} ✔")
    print(f"[mvte smoke/files] model checkpoint: {model_path} ✔")
    print("[mvte smoke/files] finished without crashing ✔")


# ---------- helpers for PYKEEN branch (optional) ----------

def make_toy_config_pykeen(root: Path) -> Path:
    """Write a minimal config_smoke_pykeen.yaml that uses data.source=pykeen."""
    cfg_path = root / "config_smoke_pykeen.yaml"
    log_dir = root / "smoke_logs_pykeen"
    out_dir = root / "smoke_checkpoints_pykeen"
    log_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    cfg_text = f"""
data:
  source: "pykeen"
  dataset: "FB15k237"

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
  num_epochs: 0         # 0 epochs: just build model/topology, then save fused V
  batch_size: 1024
  num_negatives: 2
  adversarial_temperature: 1.0
  negative_mode: "both"
  log_interval: 10

evaluation:
  batch_size_entities: 1024
  filtered: true
  hits_ks: [1, 3, 10]
  eval_every: 1

logging:
  level: "INFO"
  log_dir: "{log_dir.as_posix()}"

output:
  save_dir: "{out_dir.as_posix()}"
"""

    cfg_path.write_text(textwrap.dedent(cfg_text).lstrip(), encoding="utf-8")
    return cfg_path


def run_smoke_test_pykeen() -> None:
    """Run main_flex() once with a 'pykeen' config, if pykeen is installed.

    This just checks that:
      - the pykeen data branch is taken
      - main_flex completes end-to-end (topology, model, fused V, saving)
    """
    try:
        import pykeen  # noqa: F401
    except ImportError:
        print("[mvte smoke/pykeen] pykeen not installed, skipping pykeen smoke test.")
        return

    project_root = Path(__file__).resolve().parent
    cfg_path = make_toy_config_pykeen(project_root)

    print(f"[mvte smoke/pykeen] using config: {cfg_path}")

    sys.argv = ["main_flex.py", "--config", str(cfg_path)]
    mvte_main()

    out_dir = project_root / "smoke_checkpoints_pykeen"
    fused_path = out_dir / "fused_entity_embeddings.pt"
    model_path = out_dir / "mvte_model.pt"

    assert fused_path.exists(), f"missing fused embeddings at {fused_path}"
    assert model_path.exists(), f"missing model checkpoint at {model_path}"

    print(f"[mvte smoke/pykeen] fused embeddings: {fused_path} ✔")
    print(f"[mvte smoke/pykeen] model checkpoint: {model_path} ✔")
    print("[mvte smoke/pykeen] finished without crashing ✔")


# ---------- entrypoint ----------

def run_all_smoke_tests() -> None:
    run_smoke_test_files()
    run_smoke_test_pykeen()


if __name__ == "__main__":
    run_all_smoke_tests()
