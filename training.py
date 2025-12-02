# mvte/training.py

import logging
from typing import Optional, Tuple, Callable

import torch

from data_topology import TopologyData
from model import MVTEModel

logger = logging.getLogger(__name__)


def move_topology_to_device(
    topo: Optional[TopologyData],
    device: torch.device,
) -> Optional[TopologyData]:
    """move topology tensors to the specified device.

    performs in-place moves on the topology object and returns it.
    """
    if topo is None:
        return None

    # check if already on the target device by inspecting one tensor
    if topo.triangles.device == device:
        return topo

    topo.triangles = topo.triangles.to(device)
    topo.tetras = topo.tetras.to(device)
    topo.entity_triangle_index = topo.entity_triangle_index.to(device)
    topo.triangle_tetra_index = topo.triangle_tetra_index.to(device)
    topo.entity_tetra_index = topo.entity_tetra_index.to(device)

    return topo


def generate_negative_triples(
    pos_triples: torch.LongTensor,
    num_entities: int,
    num_negatives: int,
    device: torch.device,
    mode: str = "both",
) -> Tuple[torch.LongTensor, int]:
    """generate corrupted negative triples for a batch.

    pos_triples:
        [batch_size, 3] tensor of (h, r, t) indices.
    num_entities:
        total number of entities in the kg.
    num_negatives:
        number of negatives per positive triple.
    mode:
        - "head": corrupt head only
        - "tail": corrupt tail only
        - "both": randomly corrupt head or tail per negative

    returns:
        neg_triples_flat: [batch_size * num_negatives, 3]
        num_negatives: returned for convenience (unchanged)
    """
    if num_negatives <= 0:
        raise ValueError("num_negatives must be positive")

    batch_size = pos_triples.size(0) # we need to know how many positive triples we have in the batch

    # base tensor with repeated positives: [B, K, 3]
    neg_triples = pos_triples.unsqueeze(1).repeat(1, num_negatives, 1) 
    # repeat each positive triple K times to create a base for negatives (batch size, number of negatives, 3)

    # sample random entity ids for corruption: [B, K]
    random_entities = torch.randint(
        low=0,
        high=num_entities, # we select from the whole dataset (all entities), it's intentional to corrupt with any entity
        size=(batch_size, num_negatives),
        device=device,
    )

    if mode == "head":
        neg_triples[:, :, 0] = random_entities
    elif mode == "tail":
        neg_triples[:, :, 2] = random_entities
    elif mode == "both":
        # bernoulli mask deciding whether to corrupt head or tail
        mask = torch.rand(batch_size, num_negatives, device=device) < 0.5 # will assign True/False randomly over the corrupted set we just created (50% chance)
        # corrupt heads where mask is true
        neg_triples[:, :, 0] = torch.where( # will go element by element and pick either the random entity or the original head based on the mask
            mask, # condition
            random_entities, # if True
            neg_triples[:, :, 0], # if False
        )
        # corrupt tails where mask is false
        neg_triples[:, :, 2] = torch.where(
            ~mask, # this uses the complement of the mask, we'll corrupt the tail where head was not corrupted
            random_entities,
            neg_triples[:, :, 2],
        )
    else:
        raise ValueError(f"unknown negative sampling mode: {mode}")

    neg_triples_flat = neg_triples.view(batch_size * num_negatives, 3) # flatten from [B, K, 3] to [B * K, 3] so we have a list of all negative triples
    return neg_triples_flat, num_negatives


def self_adversarial_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    num_negatives: int,
    adversarial_temperature: float,
) -> torch.Tensor:
    """compute self-adversarial softplus loss.

    implements the negative sampling objective used in RotatE:
    - higher scores correspond to more plausible triples
    - negatives are reweighted with a softmax over their scores

    pos_scores:
        [batch_size]
    neg_scores:
        [batch_size * num_negatives]
    num_negatives:
        number of negatives per positive triple.
    adversarial_temperature:
        temperature for the softmax over negatives; larger values focus more
        on hard negatives. if non-positive, uniform weights are used.
    """
    batch_size = pos_scores.size(0)
    # derive batch_size from pos_scores to remain independent of external batch settings
    # (handles last smaller batch and ensures correct reshaping of negative scores)

    if neg_scores.numel() != batch_size * num_negatives:
        raise ValueError(
            "neg_scores size does not match batch_size * num_negatives "
            f"({neg_scores.numel()} vs {batch_size} * {num_negatives})"
        )

    # reshape negative scores back to [B, K]
    # we previously flattened them for scoring to [B*K,3] then [B*K] for scores
    neg_scores = neg_scores.view(batch_size, num_negatives)
    # this will take the first K scores for the first batch element, next K for the second, etc.

    # positive part: softplus(-s_pos)
    pos_loss = torch.nn.functional.softplus(-pos_scores)  # [B]
    # penalize low-scoring positives: softplus(-s) gives small loss for good triples, large for bad ones
    # softplus loss provides smoother gradients and leads to more stable training when combined with adversarial reweighting.

    # negative part: self-adversarial weighting over negatives
    # higher temperature means more focus on hard negatives
    if adversarial_temperature > 0.0:
        weights = torch.softmax( # softmax gives a probability distribution over the K negatives of that one positive triple
            neg_scores * adversarial_temperature,
            dim=1,
        ).detach() # no grad through weights
    else:
        # fall back to uniform weighting
        weights = torch.full_like(
            neg_scores,
            1.0 / float(num_negatives),
        )

    # weighted softplus(s_neg)
    neg_loss_per_triple = (weights * torch.nn.functional.softplus(neg_scores)).sum(
        dim=1
    )  # [B]

    # final loss: mean over triples
    loss = (pos_loss + neg_loss_per_triple).mean() # average over the batch
    return loss


def train_one_epoch(
    model: MVTEModel,
    topo: Optional[TopologyData],
    train_triples: torch.LongTensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    num_negatives: int,
    adversarial_temperature: float,
    device: torch.device,
    negative_mode: str = "both",
    log_interval: int = 100,
) -> float:
    """run one training epoch over the given triples.

    the function expects:
        - triples are already integer-encoded (h, r, t) indices.
        - model is on the target device.
        - topology tensors reside on the same device (use move_topology_to_device).

    returns:
        mean loss over all batches in the epoch (as a float).
    """
    model.train()

    # ensure triples are on the target device
    train_triples = train_triples.to(device)

    # optionally move topology to device (no-op if topo is None)
    topo = move_topology_to_device(topo, device)

    num_triples = train_triples.size(0)
    num_entities = model.entity_emb.num_embeddings

    total_loss = 0.0
    num_batches = 0

    # random permutation for shuffling
    perm = torch.randperm(num_triples, device=device) # returns a tensor containing a random permutation of integers from 0 to N-1

    for start in range(0, num_triples, batch_size): # iterate over the dataset in chunks of batch_size ( start takes values 0, batch_size, 2*batch_size, ... until we reach num_triples )
        end = min(start + batch_size, num_triples) # allows the last batch to be smaller
        batch_idx = perm[start:end] # extract the indices for this particular batch in the shuffled order
        pos_batch = train_triples[batch_idx]  # [B, 3], gathers positive triples
        batch_size_eff = pos_batch.size(0) # effective batch size (may be smaller on last batch)

        optimizer.zero_grad()

        # compute fused entity embeddings once for the batch
        _, _, V = model.get_entity_views(topo)

        # positive scores: [B]
        pos_scores = model.score_triples(pos_batch, V=V)

        # negative sampling
        neg_triples, k = generate_negative_triples(
            pos_triples=pos_batch,
            num_entities=num_entities,
            num_negatives=num_negatives,
            device=device,
            mode=negative_mode,
        )

        # negative scores: [B * K]
        neg_scores = model.score_triples(neg_triples, V=V)

        # self-adversarial loss
        loss = self_adversarial_loss(
            pos_scores=pos_scores,
            neg_scores=neg_scores,
            num_negatives=k,
            adversarial_temperature=adversarial_temperature,
        )

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

        if log_interval and (num_batches % log_interval == 0):
            logger.info(
                "batch %d | loss = %.4f | batch_size = %d",
                num_batches,
                loss.item(),
                batch_size_eff,
            )

    mean_loss = total_loss / max(num_batches, 1)
    return mean_loss


def train_model(
    model: MVTEModel,
    topo: Optional[TopologyData],
    train_triples: torch.LongTensor,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    batch_size: int,
    num_negatives: int,
    adversarial_temperature: float,
    device: Optional[torch.device] = None,
    negative_mode: str = "both",
    log_interval: int = 100,
    epoch_callback: Optional[Callable[[int, float], None]] = None,
) -> None:
    """high-level training loop over multiple epochs.

    wraps train_one_epoch and logs the mean loss per epoch.

    epoch_callback, if provided, is called as epoch_callback(epoch, mean_loss) at the end of each epoch. 
    This is where main.py can run validation. (callback avoids coupling training with validation code and a circular import)

    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    for epoch in range(1, num_epochs + 1):
        mean_loss = train_one_epoch(
            model=model,
            topo=topo,
            train_triples=train_triples,
            optimizer=optimizer,
            batch_size=batch_size,
            num_negatives=num_negatives,
            adversarial_temperature=adversarial_temperature,
            device=device,
            negative_mode=negative_mode,
            log_interval=log_interval,
        )

        logger.info(
            "epoch %d/%d | mean loss = %.4f",
            epoch,
            num_epochs,
            mean_loss,
        )
        if epoch_callback is not None:
            # Let main.py handle validation / logging / early stopping etc.
            epoch_callback(epoch, mean_loss)