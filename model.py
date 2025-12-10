# mvte/model.py

import logging
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from data_topology import TopologyData

logger = logging.getLogger(__name__)


# ---- Scoring functions -------------------------------------------------
def score_transe_from_V(
    triples: torch.LongTensor,
    V: torch.Tensor,
    R: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """TransE-style score: gamma - ||V[h] + R[r] - V[t]||_2."""
    assert triples.dim() == 2 and triples.size(1) == 3, "triples must have shape [B, 3]"

    h = triples[:, 0]
    r = triples[:, 1]
    t = triples[:, 2]
    # gets the indices for heads, relations, tails from the batch of triples
    # we then map these indices to their corresponding embeddings to get vectors for scoring
    v_h = V[h]      # [B, d]
    v_t = V[t]      # [B, d]
    r_vec = R[r]    # [B, d]

    # TransE translation: V[h] + R[r] ≈ V[t]
    diff = v_h + r_vec - v_t

    # compute negative L2 norm of the diff vector as score
    distance = torch.norm(diff, p=2, dim=-1)  # [B]
    # dim=-1 means we compute the norm across the embedding dimension for each triple in the batch (each row's d-dim vector)
    # RotatE/TransE-style score: gamma - distance
    scores = gamma - distance  # [B]
    return scores

def score_distmult_from_V(
    triples: torch.LongTensor,
    V: torch.Tensor,
    R: torch.Tensor,
    gamma: torch.Tensor | None = None,
) -> torch.Tensor:
    """DistMult-style score"""
    assert triples.dim() == 2 and triples.size(1) == 3, "triples must have shape [B, 3]"

    h = triples[:, 0]
    r = triples[:, 1]
    t = triples[:, 2]

    v_h = V[h]      # [B, d]
    v_t = V[t]      # [B, d]
    r_vec = R[r]    # [B, d]

    scores = (v_h * r_vec * v_t).sum(dim=-1)  # [B]
    return scores
# ----------------------------------------------------------------------------

SCORER_REGISTRY = {
    "transe": score_transe_from_V,
    "distmult": score_distmult_from_V,
    # later: "complex": score_complex_from_V, "rotate": score_rotate_from_V, ...
}



def _mean_aggregate(
    edge_index: torch.LongTensor,
    src: torch.Tensor,
    num_dst: int,
) -> torch.Tensor:
    """mean aggregate from src nodes to dst nodes along edges.

    edge_index:
        shape [2, num_edges]
        row 0: src node indices
        row 1: dst node indices

    src:
        [num_src, dim]

    returns:
        agg: [num_dst, dim], mean of src features per dst node
    """
    if edge_index.numel() == 0 or num_dst == 0:
        return torch.zeros(
            num_dst,
            src.size(1),
            dtype=src.dtype,
            device=src.device,
        )

    src_idx = edge_index[0]  # [num_edges]
    dst_idx = edge_index[1]  # [num_edges]

    dim = src.size(1)

    agg = torch.zeros( # allocate an empty accumulator for dst nodes
        num_dst,
        dim,
        dtype=src.dtype,
        device=src.device,
    ) # will eventually contain the sum of all messages arriving at destination node d
    agg.index_add_(0, dst_idx, src[src_idx])
    # src[src_idx] extracts the feature vectors of the source nodes of each edge
    # dst_idx tells where each of these feature vectors should be added
    # index_add_ accumulates these vectors into the agg tensor at the appropriate dst indices, 
    # conceptually: we have an edge between dst_idx[i] and src_idx[i], so we add the feature vector of src_idx[i] to agg[dst_idx[i]] (the higher dimensional entity it belongs to)

    counts = torch.zeros(
        num_dst,
        dtype=src.dtype,
        device=src.device,
    )
    # initialize a counts tensor to count the number of incoming messages for each dst node, which is why we use num_dst

    ones = torch.ones(
        src_idx.size(0),
        dtype=src.dtype,
        device=src.device,
    )
    # create a tensor of ones with the same length as the number of edges
    counts.index_add_(0, dst_idx, ones)
    # accumulates ones into the counts tensor such that everytime an edge points to a dst node, we increment its (dst node) count by 1 
    # we then essentially have the number of incoming messages for each dst node
    # this can be hard coded to be 3 or 4 for triangles/tetras, but this is more general and flexible in case of future changes

    mask = counts > 0 # the mask tells use which dst nodes received at least one message ,but this is always true by construction in our case, but good to be safe in case of manipulations mid way
    if mask.any():
        agg[mask] = agg[mask] / counts[mask].unsqueeze(-1) # we avoid division by zero and unsqueeze to match dimensions

    return agg


class EntityToTriangleGNN(nn.Module):
    """gnn2: entity -> triangle message passing on entity–triangle bipartite graph."""

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim

        self.lin = nn.Linear(dim, hidden_dim) # input layer must be of dim size because tri_agg outputs num_triangles x dim 
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        entity_emb: torch.Tensor,              # [num_entities, dim]
        entity_triangle_index: torch.Tensor,   # [2, num_edges], row0: entity, row1: triangle
        num_triangles: int,
    ) -> torch.Tensor:
        """compute triangle embeddings from entity embeddings."""
        tri_agg = _mean_aggregate(
            edge_index=entity_triangle_index,
            src=entity_emb,
            num_dst=num_triangles,
        ) # [num_triangles, dim]
        tri_out = self.lin(tri_agg)
        tri_out = self.act(tri_out)
        tri_out = self.dropout(tri_out)
        return tri_out # [num_triangles, hidden_dim]


class TriangleToTetraGNN(nn.Module):
    """gnn3: triangle -> tetra message passing on triangle–tetra bipartite graph."""

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim

        self.lin = nn.Linear(dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tri_emb: torch.Tensor,               # [num_triangles, dim]
        triangle_tetra_index: torch.Tensor,  # [2, num_edges], row0: triangle, row1: tetra
        num_tetras: int,
    ) -> torch.Tensor:
        """compute tetra embeddings from triangle embeddings."""
        tet_agg = _mean_aggregate(
            edge_index=triangle_tetra_index,
            src=tri_emb,
            num_dst=num_tetras,
        )
        tet_out = self.lin(tet_agg)
        tet_out = self.act(tet_out)
        tet_out = self.dropout(tet_out)
        return tet_out

class TopologyEncoder(nn.Module):
    """topology operator T(E): entities -> triangles -> tetras -> entities.

    given base entity embeddings E and precomputed topology data, this module
    returns V_topo, a topology-aware entity embedding of the same shape as E.
    """

    # def __init__(
    #     self,
    #     dim: int,
    #     dropout: float = 0.0,
    # ):
    #     super().__init__()
    #     self.gnn2 = EntityToTriangleGNN(dim=dim, hidden_dim=dim, dropout=dropout)
    #     self.gnn3 = TriangleToTetraGNN(dim=dim, hidden_dim=dim, dropout=dropout)

    # opted for a more flexible approach with separate hidden dims for tri/tet gnn layers
    def __init__(
        self,
        input_dim: int,
        tri_hidden_dim: int,
        tet_hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # entity -> triangle
        self.gnn2 = EntityToTriangleGNN(
            dim=input_dim,
            hidden_dim=tri_hidden_dim,
            dropout=dropout,
        )

        # triangle -> tetra
        self.gnn3 = TriangleToTetraGNN(
            dim=tri_hidden_dim,       # <- input is gnn2's output
            hidden_dim=tet_hidden_dim,
            dropout=dropout,
        )

        # simple projection: tet_dim -> entity_dim so we can aggregate to V_topo and have it be the same dim as entity_emb
        self.tet_to_entity = nn.Linear(tet_hidden_dim, input_dim)

    def forward(
        self,
        entity_emb: torch.Tensor,  # [num_entities, dim]
        topo: TopologyData,
    ) -> torch.Tensor:
        """compute V_topo given base entity embeddings and topology data."""
        num_entities = topo.num_entities
        num_triangles = topo.triangles.size(0)
        num_tetras = topo.tetras.size(0)

        if num_triangles == 0 or num_tetras == 0:
            logger.debug("no triangles or tetrahedra; returning zero topology embedding")
            return torch.zeros_like(entity_emb)

        # entity -> triangle
        tri_emb = self.gnn2(
            entity_emb=entity_emb,
            entity_triangle_index=topo.entity_triangle_index,
            num_triangles=num_triangles,
        )

        # triangle -> tetra
        tet_emb = self.gnn3(
            tri_emb=tri_emb,
            triangle_tetra_index=topo.triangle_tetra_index,
            num_tetras=num_tetras,
        ) # [num_tetras, tet_hidden_dim]
        # tetra -> entity dimension projection
        tet_proj = self.tet_to_entity(tet_emb)  # [num_tetras, entity_dim]

        # tetra -> entity (downward aggregation)
        # topo.entity_tetra_index stores edges (entity, tetra)
        # for downward message passing we need edges (tetra -> entity)
        et_index = topo.entity_tetra_index
        if et_index.numel() == 0:
            logger.debug("no entity–tetra incidences; returning zero topology embedding")
            return torch.zeros_like(entity_emb)

        te_index = torch.stack(
            [et_index[1], et_index[0]],
            dim=0,
        )  # row0: tetra, row1: entity

        v_topo = _mean_aggregate(
            edge_index=te_index,
            src=tet_proj,
            num_dst=num_entities,
        )

        return v_topo


class MVTEModel(nn.Module):
    """multi-view topology-aware kg embedding model with translational scoring.

    components:
        - entity embedding table E
        - relation embedding table R
        - topology encoder T(E) using triangles/tetrahedra
        - fusion: V = w1 * E + w2 * V_topo
        - score: s(h, r, t) = -||V[h] + R[r] - V[t]||_2
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        tri_hidden_dim: int,
        tet_hidden_dim: int,
        dropout: float = 0.0,
        gamma: float = 12.0,   # <- new for better scoring, default similar to RotatE

    ):
        super().__init__()

        # this is for bookkeeeping purposes to be stored
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        # margin parameter γ (kept fixed, like RotatE)
        self.gamma = nn.Parameter(
            torch.tensor([gamma], dtype=torch.float),
            requires_grad=False,
        )
        self.fusion_mode = "learned"  # default mode; can be changed externally
        
        # base scoring function: "transe" (default) or "distmult"
        # can be changed externally, e.g. model.base_scorer = "distmult"
        self.base_scorer: str = "transe"

        # base embedding tables
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)

        # topology module T(E)
        self.topology_encoder = TopologyEncoder(
            input_dim=embedding_dim,
            tri_hidden_dim=tri_hidden_dim,
            tet_hidden_dim=tet_hidden_dim,
            dropout=dropout,
        )

        # fusion parameters α, softmax -> (w1, w2)
        # initialised to zero so w1 = w2 = 0.5 at start in the forward pass
        self.fusion_alpha = nn.Parameter(torch.zeros(2, dtype=torch.float))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """xavier init for embeddings and gnn layers."""
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_fusion_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """return (w1, w2) from softmax over fusion_alpha."""
        mode = getattr(self, "fusion_mode", "learned")
        if mode == "topo_only":
            # use only topology-aware embeddings
            return(
                torch.tensor(0.0, device=self.fusion_alpha.device),
                torch.tensor(1.0, device=self.fusion_alpha.device),
            )
        if mode == "equal":
            # equal weights
            return(
                torch.tensor(0.5, device=self.fusion_alpha.device),
                torch.tensor(0.5, device=self.fusion_alpha.device),
            )
        if mode == "custom":
            # custom fixed weights set externally
            return(
                torch.tensor(0.7, device=self.fusion_alpha.device),
                torch.tensor(0.3, device=self.fusion_alpha.device),
            )
        # default: learned weights via softmax
        weights = F.softmax(self.fusion_alpha, dim=0)
        w1 = weights[0]
        w2 = weights[1]
        return w1, w2 # will be 0.5, 0.5 at the start of training

    def get_entity_views(
        self,
        topo: Optional[TopologyData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """return (E, V_topo, V) entity embeddings.

        E:
            base entity embeddings from the lookup table
        V_topo:
            topology-aware entity embeddings T(E); zero if topo is None
        V:
            fused entity embeddings w1 * E + w2 * V_topo
        """
        E = self.entity_emb.weight

        if topo is None:
            V_topo = torch.zeros_like(E)
        else:
            V_topo = self.topology_encoder(E, topo)

        w1, w2 = self.get_fusion_weights()
        V = w1 * E + w2 * V_topo

        return E, V_topo, V
    
    def _score_from_V(
        self,
        triples: torch.LongTensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """Scores given a precomputed entity view V.
        Delegates to a base scoring function selected by `self.base_scorer`."""
        assert triples.dim() == 2 and triples.size(1) == 3, "triples must have shape [B, 3]"

        R = self.relation_emb.weight  # [num_relations, d]

        scorer_fn = SCORER_REGISTRY.get(self.base_scorer, None)
        if scorer_fn is None:
            raise ValueError(f"unknown base_scorer '{self.base_scorer}'")
        return scorer_fn(
            triples=triples,
            V=V,
            R=R,
            gamma=self.gamma,
        )
    
    # the function to score from a precomputed V is useful to compute V once per batch and use it for both positive and negative triples without recomputing

    # it is then used inside the score_triples function below
    def score_triples(
        self,
        triples: torch.LongTensor,
        topo: Optional[TopologyData] = None,
        V: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """compute TransE-style scores for a batch of triples.

        triples:
            LongTensor of shape [batch_size, 3] with (h, r, t) indices.
        topo:
            topology data built from training triples. if None, no topology is used.
        V: 
            optional precomputed fused entity embeddings [num_entities, d].
            If provided, topo is ignored and V is used directly.
        returns:
            scores: [batch_size], higher is better.
        """
        assert triples.dim() == 2 and triples.size(1) == 3, "triples must have shape [B, 3]"

        if V is None:
            # default behavior: compute entity views (including topology) here
            _, _, V = self.get_entity_views(topo) # we retrieve the fused entity embeddings V which is all we need for scoring

        return self._score_from_V(triples, V)


    def forward(
        self,
        triples: torch.LongTensor,
        topo: Optional[TopologyData] = None,
    ) -> torch.Tensor:
        """alias for score_triples to integrate with training loops."""
        return self.score_triples(triples, topo)

"""
Note on forward(), score_triples(), and training-time caching
-------------------------------------------------------------

The model provides two scoring paths:

1) forward(triples, topo)
   - Simple, default API.
   - Internally calls score_triples(triples, topo).
   - Topology (get_entity_views) is computed *inside* the call.
   - Useful for quick experiments, debugging, or when performance is not critical.

2) score_triples(triples, topo=None, V=None)
   - If V is None:
         Topology encoder is executed and V is computed internally.
   - If a precomputed V is provided:
         Topology computation is skipped and the provided V is used directly.

During training, the recommended usage is:
    _, _, V = model.get_entity_views(topo)   # compute fused embeddings once per batch
    pos_scores = model.score_triples(pos_batch, V=V)
    neg_scores = model.score_triples(neg_batch, V=V)

This avoids recomputing the topology encoder multiple times per batch,
while maintaining correct gradients. The forward() method remains as a
clean, idiomatic PyTorch entry point, but the training script should use the explicit, 
optimized path through score_triples(..., V=V).
"""
