# mvte/evaluation.py

import logging
from typing import Dict, Optional, Tuple, Sequence

import torch

from data_topology import TopologyData
from model import MVTEModel
from training import move_topology_to_device

logger = logging.getLogger(__name__)


def _build_all_triples_set(
    triples: torch.LongTensor,
) -> "set[Tuple[int, int, int]]":
    """Convert a [N, 3] triple tensor to a Python set of (h, r, t) tuples.

    Used for filtered evaluation to identify other true triples.
    """
    assert triples.dim() == 2 and triples.size(1) == 3, "triples must have shape [N, 3]"
    triples_list = triples.tolist()
    return set((h, r, t) for (h, r, t) in triples_list)


def _compute_side_rank(
    model: MVTEModel,
    V: torch.Tensor,
    triple: torch.LongTensor,
    all_true_triples: Optional["set[Tuple[int, int, int]]"],
    side: str,
    device: torch.device,
    batch_size_entities: int,
    filtered: bool,
) -> int:
    """Compute rank for predicting head or tail of a single triple.

    side:
        "head" or "tail"
    triple:
        [3] tensor (h, r, t) on *any* device (will be moved to CPU for set ops).
    V:
        fused entity embeddings [num_entities, d] on `device`.
    """
    assert side in {"head", "tail"}, "side must be 'head' or 'tail'"

    num_entities = V.size(0)
    h, r, t = triple.tolist()  # small, fine on CPU

    if side == "head":
        true_idx = h
    else:
        true_idx = t

    # We will compute scores for *all* entities as candidate heads/tails.
    # To control memory, we process entities in chunks.
    all_scores_chunks = []

    entity_ids = torch.arange(num_entities, device=device, dtype=torch.long)

    for start in range(0, num_entities, batch_size_entities):
        end = min(num_entities, start + batch_size_entities)
        batch_entities = entity_ids[start:end]

        if side == "head":
            batch_heads = batch_entities
            batch_rels = torch.full_like(batch_heads, r)
            batch_tails = torch.full_like(batch_heads, t)
        else:  # side == "tail"
            batch_heads = torch.full_like(batch_entities, h)
            batch_rels = torch.full_like(batch_entities, r)
            batch_tails = batch_entities

        batch_triples = torch.stack([batch_heads, batch_rels, batch_tails], dim=1)
        # scores: [batch_size_entities] (or smaller for last chunk)
        with torch.no_grad():
            batch_scores = model.score_triples(batch_triples, topo=None, V=V)

        if filtered and all_true_triples is not None:
            # Mask out *other* true triples from the candidate set.
            # We keep the current (h, r, t) triple unmasked.
            batch_triples_cpu = batch_triples.cpu().tolist()

            mask = []
            for (hh, rr, tt) in batch_triples_cpu:
                if (hh, rr, tt) == (h, r, t):
                    # this is the test triple itself -> do not filter
                    mask.append(False)
                elif (hh, rr, tt) in all_true_triples:
                    # another true triple -> filter it out
                    mask.append(True)
                else:
                    mask.append(False)

            mask_tensor = torch.tensor(mask, device=device, dtype=torch.bool)
            batch_scores = batch_scores.masked_fill(mask_tensor, float("-inf"))

        # keep scores on device
        all_scores_chunks.append(batch_scores) # previously (batch_scores.cpu())

    # Concatenate all candidate scores into [num_entities]
    all_scores = torch.cat(all_scores_chunks, dim=0)
    assert all_scores.shape[0] == num_entities

    true_score = all_scores[true_idx].item()

    # Rank = 1 + number of candidates with strictly higher score.
    # (higher score = better triple)
    better_count = (all_scores > true_score).sum().item() # sum on GPU, then .item()
    rank = int(better_count) + 1

    return rank


def compute_ranks(
    model: MVTEModel,
    topo: Optional[TopologyData],
    eval_triples: torch.LongTensor,
    all_triples: Optional[torch.LongTensor] = None,
    device: Optional[torch.device] = None,
    V: Optional[torch.Tensor] = None,
    batch_size_entities: int = 1024,
    filtered: bool = True,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Compute head and tail ranks for a batch of evaluation triples.

    Args
    ----
    model:
        MVTEModel instance (will be moved to `device`).
    topo:
        TopologyData built from training triples. Used only if V is None.
    eval_triples:
        [N, 3] tensor of (h, r, t) to evaluate.
    all_triples:
        [M, 3] tensor of all known true triples (train + valid + test) for *filtered*
        setting. If None, only eval_triples are used for filtering.
    device:
        torch.device to use. If None, uses CUDA if available, else CPU.
    V:
        Optional precomputed fused entity embeddings [num_entities, d].
        If provided, `topo` is ignored here.
    batch_size_entities:
        Chunk size for enumerating candidate entities during ranking.
        Controls memory/speed trade-off.
    filtered:
        If True, perform *filtered* evaluation (Bordes et al.): other true triples
        are removed from the candidate set before ranking.

    Returns
    -------
    head_ranks: [N] tensor of head ranks.
    tail_ranks: [N] tensor of tail ranks.
    """
    assert eval_triples.dim() == 2 and eval_triples.size(1) == 3, "eval_triples must have shape [N, 3]"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Compute fused entity embeddings V if not provided.
    if V is None:
        if topo is not None:
            topo_on_device = move_topology_to_device(topo, device)
        else:
            topo_on_device = None

        with torch.no_grad():
            _, _, V = model.get_entity_views(topo_on_device)
    else:
        V = V.to(device)

    num_entities = V.size(0)
    logger.info(
        "computing ranks for %d triples | num_entities=%d | filtered=%s",
        eval_triples.size(0),
        num_entities,
        filtered,
    )

    # Build all-true set for filtered evaluation, on CPU.
    all_true_triples_set: Optional["set[Tuple[int, int, int]]"] = None
    if filtered:
        if all_triples is None:
            all_triples = eval_triples
        all_true_triples_set = _build_all_triples_set(all_triples.cpu())

    head_ranks = []
    tail_ranks = []

    for idx in range(eval_triples.size(0)):
        triple = eval_triples[idx]

        head_rank = _compute_side_rank(
            model=model,
            V=V,
            triple=triple,
            all_true_triples=all_true_triples_set,
            side="head",
            device=device,
            batch_size_entities=batch_size_entities,
            filtered=filtered,
        )

        tail_rank = _compute_side_rank(
            model=model,
            V=V,
            triple=triple,
            all_true_triples=all_true_triples_set,
            side="tail",
            device=device,
            batch_size_entities=batch_size_entities,
            filtered=filtered,
        )

        head_ranks.append(head_rank)
        tail_ranks.append(tail_rank)

        if (idx + 1) % 100 == 0 or idx == eval_triples.size(0) - 1:
            logger.info(
                "evaluated %d / %d triples",
                idx + 1,
                eval_triples.size(0),
            )

    head_ranks_tensor = torch.tensor(head_ranks, dtype=torch.long)
    tail_ranks_tensor = torch.tensor(tail_ranks, dtype=torch.long)

    return head_ranks_tensor, tail_ranks_tensor


def compute_link_prediction_metrics(
    head_ranks: torch.LongTensor,
    tail_ranks: torch.LongTensor,
    hits_ks: Sequence[int] = (1, 3, 10),
) -> Dict[str, float]:
    """Compute standard link prediction metrics from per-triple ranks.

    Metrics (over both head and tail predictions):
      - MR:   mean rank
      - MRR:  mean reciprocal rank
      - Hits@k for each k in hits_ks

    Args
    ----
    head_ranks, tail_ranks:
        [N] tensors with 1-based ranks for each triple.
    hits_ks:
        List/tuple of k values for Hits@k.

    Returns
    -------
    metrics:
        dict with keys: "MR", "MRR", "Hits@1", "Hits@3", "Hits@10", ...
    """
    assert head_ranks.shape == tail_ranks.shape, "head_ranks and tail_ranks must have same shape"

    all_ranks = torch.cat([head_ranks, tail_ranks], dim=0).float()  # [2N]

    mr = all_ranks.mean().item()
    mr_std = all_ranks.std(unbiased=False).item() # compute standard deviation, unbiased=False for population std


    reciprocal_ranks = 1.0 / all_ranks
    mrr  = reciprocal_ranks.mean().item()
    mrr_std = reciprocal_ranks.std(unbiased=False).item()


    metrics: Dict[str, float] = {
        "MR": mr,
        "MR_std": mr_std,
        "MRR": mrr,
        "MRR_std": mrr_std,
    }

    for k in hits_ks:
        hits_k = (all_ranks <= float(k)).float().mean().item()
        metrics[f"Hits@{k}"] = hits_k

    return metrics


def evaluate_link_prediction(
    model: MVTEModel,
    topo: Optional[TopologyData],
    eval_triples: torch.LongTensor,
    all_triples: Optional[torch.LongTensor] = None,
    device: Optional[torch.device] = None,
    V: Optional[torch.Tensor] = None,
    batch_size_entities: int = 1024,
    filtered: bool = True,
    hits_ks: Sequence[int] = (1, 3, 10),
) -> Dict[str, float]:
    """Convenience wrapper: ranks + metrics for a set of triples.

    This is the main entry point you can call from:
      - training loop (for validation set), and
      - after training (for test set).

    Args
    ----
    model, topo, eval_triples, all_triples, device, V, batch_size_entities, filtered:
        See `compute_ranks` docstring.
    hits_ks:
        Values of k for Hits@k.

    Returns
    -------
    metrics:
        dict with MR, MRR, Hits@k, computed over both head and tail predictions.
    """
    head_ranks, tail_ranks = compute_ranks(
        model=model,
        topo=topo,
        eval_triples=eval_triples,
        all_triples=all_triples,
        device=device,
        V=V,
        batch_size_entities=batch_size_entities,
        filtered=filtered,
    )

    metrics = compute_link_prediction_metrics(
        head_ranks=head_ranks,
        tail_ranks=tail_ranks,
        hits_ks=hits_ks,
    )

    logger.info(
        "evaluation metrics: " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    )

    return metrics
