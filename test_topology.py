import torch
from pykeen.datasets import FB15k237

from data_topology import build_topology_from_triples


def main():
    print("Loading FB15k-237...")
    ds = FB15k237()

    # use train split for topology extraction
    triples = ds.training.mapped_triples
    num_entities = ds.num_entities

    print(f"Triples loaded: {triples.shape[0]}")
    print(f"Num entities:  {num_entities}")

    print("\nBuilding topology...")
    topo = build_topology_from_triples(
        num_entities=num_entities,
        triples=triples,
        max_triangles_per_entity=None,
        max_tetras_per_entity=None,
    )

    print("\n=== Topology Diagnostics ===")
    print(f"Triangles: {topo.triangles.size(0)}")
    print(f"Tetras:    {topo.tetras.size(0)}")

    print(f"Entity–Triangle edges: {topo.entity_triangle_index.size(1)}")
    print(f"Triangle–Tetra edges:  {topo.triangle_tetra_index.size(1)}")
    print(f"Entity–Tetra edges:    {topo.entity_tetra_index.size(1)}")

    print("\nShapes:")
    print(f"triangles:              {tuple(topo.triangles.shape)}")
    print(f"tetras:                 {tuple(topo.tetras.shape)}")
    print(f"entity_triangle_index:  {tuple(topo.entity_triangle_index.shape)}")
    print(f"triangle_tetra_index:   {tuple(topo.triangle_tetra_index.shape)}")
    print(f"entity_tetra_index:     {tuple(topo.entity_tetra_index.shape)}")

    if topo.simplex_tree is None:
        print("\n[Warning] Gudhi not available: using fallback (no closure).")

    print("\nDone. Looks good if no errors occurred.")


if __name__ == "__main__":
    main()
