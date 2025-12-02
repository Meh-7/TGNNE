"""
Minimal smoke test for evaluation.py

Run:
    python test_eval.py
"""

import torch

# import functions under test
from evaluation import (
    compute_ranks,
    compute_link_prediction_metrics,
    evaluate_link_prediction,
)
from data_topology import TopologyData  # just to ensure import works
# MVTEModel type is only for hints in evaluation.py; we don't need it here.


# -----------------------------
# Dummy model for evaluation
# -----------------------------
class DummyModel(torch.nn.Module):
    """
    Minimal stub to exercise evaluation code.
    It overrides score_triples and get_entity_views.
    """

    def __init__(self, num_entities: int, num_relations: int, dim: int = 8):
        super().__init__()  # IMPORTANT: initialize nn.Module

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        # trivial embeddings
        self.E = torch.nn.Embedding(num_entities, dim)
        self.R = torch.nn.Embedding(num_relations, dim)

    def score_triples(self, triples, topo=None, V=None):
        """
        Simple fake scoring function:
          score = -||E[h] + R[r] - E[t]|| (TransE-style but meaningless here)
        """
        h = triples[:, 0]
        r = triples[:, 1]
        t = triples[:, 2]

        Eh = self.E(h)
        Er = self.R(r)
        Et = self.E(t)

        score = -(Eh + Er - Et).norm(dim=-1)
        return score

    def get_entity_views(self, topo):
        """
        Return (E_raw, E_topo, V_fused).
        For a smoke test we just return the same tensor three times.
        """
        V = self.E.weight.clone()
        return V, V, V


def main():
    num_entities = 5
    num_relations = 2

    eval_triples = torch.tensor(
        [
            [0, 0, 1],
            [2, 1, 3],
            [4, 0, 0],
        ],
        dtype=torch.long,
    )

    all_triples = torch.tensor(
        [
            [0, 0, 1],
            [2, 1, 3],
            [4, 0, 0],
            [1, 1, 4],
        ],
        dtype=torch.long,
    )

    model = DummyModel(num_entities, num_relations)
    V = model.E.weight.clone()

    # ---- compute_ranks ----
    head_ranks, tail_ranks = compute_ranks(
        model=model,
        topo=None,            # ignored because V is provided
        eval_triples=eval_triples,
        all_triples=all_triples,
        device=None,
        V=V,
        batch_size_entities=5,
        filtered=True,
    )

    print("compute_ranks OK")
    print("head_ranks:", head_ranks)
    print("tail_ranks:", tail_ranks)
    assert head_ranks.shape == (eval_triples.size(0),)
    assert tail_ranks.shape == (eval_triples.size(0),)

    # ---- compute_link_prediction_metrics ----
    metrics = compute_link_prediction_metrics(
        head_ranks=head_ranks,
        tail_ranks=tail_ranks,
        hits_ks=(1, 3, 10),
    )
    print("compute_link_prediction_metrics OK")
    print(metrics)
    assert "MR" in metrics and "MRR" in metrics

    # ---- evaluate_link_prediction (full pipeline) ----
    metrics2 = evaluate_link_prediction(
        model=model,
        topo=None,
        eval_triples=eval_triples,
        all_triples=all_triples,
        device=None,
        V=V,
        batch_size_entities=5,
        filtered=True,
        hits_ks=(1, 3, 10),
    )
    print("evaluate_link_prediction OK")
    print(metrics2)

    print("\nAll smoke tests passed!")


if __name__ == "__main__":
    main()
