import torch
from model import MVTEModel
from data_topology import build_topology_from_triples
from training import (
    move_topology_to_device,
    generate_negative_triples,
    self_adversarial_loss,
)

# ---------------------------------------------------
# 1. Create tiny fake KG triples
# ---------------------------------------------------
#    (h, r, t)
triples = torch.tensor([
    [0, 0, 1],
    [1, 0, 2],
    [2, 0, 3],
], dtype=torch.long)

num_entities = 4
num_relations = 1
embedding_dim = 16

print("=== Building topology from triples ===")
topo = build_topology_from_triples(
    triples=triples,
    num_entities=num_entities,
    max_triangles_per_entity=10,
    max_tetras_per_entity=10,
)

# ---------------------------------------------------
# 2. Build model
# ---------------------------------------------------
model = MVTEModel(
    num_entities=num_entities,
    num_relations=num_relations,
    embedding_dim=embedding_dim,
    tri_hidden_dim=32,
    tet_hidden_dim=32,
    dropout=0.1,
    gamma=12.0,  # new scoring margin
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
triples = triples.to(device)
topo = move_topology_to_device(topo, device)

# ---------------------------------------------------
# 3. Compute V once (as training does per batch)
# ---------------------------------------------------
E, V_topo, V = model.get_entity_views(topo)
print("Entity embeddings E:", E.shape)
print("Topology views V_topo:", V_topo.shape)
print("Fused V:", V.shape)

# ---------------------------------------------------
# 4. Positive scores
# ---------------------------------------------------
pos_scores = model.score_triples(triples, V=V)
print("Positive scores:", pos_scores)

# ---------------------------------------------------
# 5. Generate negatives & score them
# ---------------------------------------------------
neg_triples, K = generate_negative_triples(
    pos_triples=triples,
    num_entities=num_entities,
    num_negatives=4,
    device=device,
    mode="both",
)

neg_scores = model.score_triples(neg_triples, V=V)
print("Negative scores:", neg_scores[:10])  # show only first few

# ---------------------------------------------------
# 6. Compute loss
# ---------------------------------------------------
loss = self_adversarial_loss(
    pos_scores=pos_scores,
    neg_scores=neg_scores,
    num_negatives=K,
    adversarial_temperature=1.0,
)

print("Loss:", loss.item())

# ---------------------------------------------------
# 7. Backprop test
# ---------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("✓ Backward + step completed with no errors.")
print("=== TEST FINISHED SUCCESSFULLY ===")
