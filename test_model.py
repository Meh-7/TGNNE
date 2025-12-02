import torch
from model import MVTEModel
from data_topology import TopologyData

def build_dummy_topology():
    """
    Constructs a tiny, valid TopologyData instance:
        entities: 0,1,2
        triangle: (0,1,2)
        tetra: (0,1,2,0)  # dummy wrap-around just to create one tetra
    """
    num_entities = 3

    # One triangle (0,1,2)
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.long)

    # One tetra (fake but valid shape): (0,1,2,0)
    tetras = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)

    # Incidence: entity → triangle
    entity_triangle_edges = [
        (0, 0), (1, 0), (2, 0)
    ]
    entity_triangle_index = torch.tensor(entity_triangle_edges).t()

    # Incidence: triangle → tetra
    triangle_tetra_edges = [
        (0, 0)
    ]
    triangle_tetra_index = torch.tensor(triangle_tetra_edges).t()

    # Incidence: entity → tetra
    entity_tetra_edges = [
        (0, 0), (1, 0), (2, 0)
    ]
    entity_tetra_index = torch.tensor(entity_tetra_edges).t()

    topo = TopologyData(
        num_entities=num_entities,
        triangles=triangles,
        tetras=tetras,
        entity_triangle_index=entity_triangle_index,
        triangle_tetra_index=triangle_tetra_index,
        entity_tetra_index=entity_tetra_index,
    )
    return topo


def main():
    # ----- Build dummy topology -----
    topo = build_dummy_topology()

    # ----- Instantiate model -----
    model = MVTEModel(
        num_entities=3,
        num_relations=2,
        embedding_dim=16,
        tri_hidden_dim=16,
        tet_hidden_dim=16,
        dropout=0.1,
    )

    print("Model loaded successfully.")

    # ----- Test get_entity_views -----
    E, V_topo, V = model.get_entity_views(topo)
    print("E shape:", E.shape)
    print("V_topo shape:", V_topo.shape)
    print("V shape:", V.shape)

    # ----- Test scoring -----
    triples = torch.tensor([
        [0, 0, 1],
        [1, 1, 2],
    ], dtype=torch.long)

    scores = model(triples, topo=topo)
    print("Scores:", scores)

    # ----- Test backward-pass -----
    loss = -scores.mean()
    loss.backward()
    print("Backward pass successful. Gradients exist:",
          model.entity_emb.weight.grad is not None)


if __name__ == "__main__":
    main()
