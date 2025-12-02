# mvte/data_topology_networkx.py

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import torch

logger = logging.getLogger(__name__)


@dataclass
class TopologyData:
    """container for simplicial views and bipartite incidence tensors.

    all indices are assumed to be 0-based integer ids.
    """

    num_entities: int

    # simplices in terms of entity ids
    triangles: torch.LongTensor  # [num_triangles, 3]
    tetras: torch.LongTensor  # [num_tetras, 4]

    # bipartite incidence edge indices (shape [2, num_edges])
    entity_triangle_index: torch.LongTensor  # row 0: entity, row 1: triangle
    triangle_tetra_index: torch.LongTensor  # row 0: triangle, row 1: tetra
    entity_tetra_index: torch.LongTensor  # row 0: entity, row 1: tetra

    # optional gudhi simplex tree for centralized simplicial complex storage
    simplex_tree: Optional[Any] = None


def build_1_skeleton(num_entities: int, triples: torch.LongTensor,) -> torch.LongTensor:
    """build undirected 1-skeleton edge_index from triples.

    edges are between entities that co-occur in any triple.
    relation types are ignored.
    self-loops are removed and duplicate edges are merged.
    edge_index has shape [2, num_edges] with undirected edges stored once (u, v) with u < v
    """
    assert triples.dim() == 2 and triples.size(1) == 3, "triples must have shape [N, 3]"
    heads = triples[:, 0]
    tails = triples[:, 2]
    # collect undirected edges (u, v) with u < v
    u = torch.minimum(heads, tails)
    v = torch.maximum(heads, tails)
    # this makes sure edges are undirected: (3, 1) and (1, 3) both yield (1, 3)
    edges = torch.stack([u, v], dim=1)

    # remove self-loops if any
    mask = edges[:, 0] != edges[:, 1]
    edges = edges[mask]

    if edges.numel() == 0:
        logger.warning("no edges found when building 1-skeleton")
        return edges.t().contiguous()

    # deduplicate edges
    edges_unique = torch.unique(edges, dim=0)

    if edges_unique.max().item() >= num_entities:
        raise ValueError("edge index exceeds num_entities")

    edge_index = edges_unique.t().contiguous()
    # transpose to shape [2, num_edges] because that is the format expected by gnns
    logger.info(
        "built 1-skeleton with %d entities and %d undirected edges",
        num_entities,
        edge_index.size(1),
    )
    return edge_index


def _enumerate_triangles_and_tetras_from_graph(
    g: nx.Graph,
    max_triangles_per_entity: Optional[int] = None,
    max_tetras_per_entity: Optional[int] = None,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int, int]]]:
    """enumerate 3- and 4-cliques (triangles and tetrahedra) only.

    enumeration is ordered by clique size, so once k > 4 appears,
    no further 3- or 4-cliques exist and the loop breaks early.

    should avoid using max_per_entity caps, they can break the logic and chain of faces.
    logic can be fixed for that by enforcing face inclusion, but currently not implemented.
    enforcing that proper logic can also help by building face inclusion during enumeration,
    instead of relying on gudhi simplex tree to do that later.
    """

    triangles_set = set()
    tetras_set = set()

    tri_counts: Dict[int, int] = {}  # tracks how many triangles each entity participates in
    tet_counts: Dict[int, int] = {}  # tracks how many tetras each entity participates in

    # enumerate all cliques in the graph in order of size
    for clique in nx.enumerate_all_cliques(g):
        k = len(clique)

        # ignore 1-cliques and 2-cliques (single nodes and edges)
        if k < 3:
            continue

        # k > 4 → no more 3- or 4-cliques will appear
        if k > 4:
            break

        # ---- triangles ----
        if k == 3:
            tri = tuple(sorted(clique))
            if _accept_simplex(tri, tri_counts, max_triangles_per_entity):
                triangles_set.add(tri)

        # ---- tetrahedra ----
        elif k == 4:
            tet = tuple(sorted(clique))
            if _accept_simplex(tet, tet_counts, max_tetras_per_entity):
                tetras_set.add(tet)

            # no need to add 3-faces explicitly:
            # enumerate_all_cliques has already yielded the 3-cliques

    # sort for deterministic output
    triangles = sorted(triangles_set)
    tetras = sorted(tetras_set)

    return triangles, tetras


def _accept_simplex(
    simplex: Sequence[int],
    counts: Dict[int, int],
    max_per_entity: Optional[int],
) -> bool:
    """decide whether to accept a simplex under per-entity cap."""
    if max_per_entity is None:
        for v in simplex:
            counts[v] = counts.get(v, 0) + 1
        return True

    for v in simplex:
        if counts.get(v, 0) >= max_per_entity:
            return False

    for v in simplex:
        counts[v] = counts.get(v, 0) + 1
    return True


def extract_simplicial_structure(
    num_entities: int,
    edge_index: torch.LongTensor,
    max_triangles_per_entity: Optional[int] = None,
    max_tetras_per_entity: Optional[int] = None,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int, int]]]:
    """extract triangles and tetrahedra from the 1-skeleton graph.

    uses networkx enumerate_all_cliques on the undirected 1-skeleton.
    caps the number of triangles / tetras per entity if requested.

    returns:
        triangles_list: list of (v0, v1, v2) tuples
        tetras_list: list of (v0, v1, v2, v3) tuples
    """
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index must have shape [2, E]"

    g = nx.Graph()
    g.add_nodes_from(range(num_entities))

    # edge_index is undirected edges with u < v
    edges = edge_index.t().tolist()
    # convert to list of (u, v) pairs for networkx (requires python-native sequences)
    g.add_edges_from(edges)

    logger.info(
        "running clique enumeration on graph with %d nodes and %d edges",
        g.number_of_nodes(),
        g.number_of_edges(),
    )

    triangles_list, tetras_list = _enumerate_triangles_and_tetras_from_graph(
        g,
        max_triangles_per_entity=max_triangles_per_entity,
        max_tetras_per_entity=max_tetras_per_entity,
    )

    logger.info(
        "enumerated simplices: %d triangles and %d tetrahedra (pre-simplex-tree)",
        len(triangles_list),
        len(tetras_list),
    )

    return triangles_list, tetras_list


def build_simplex_tree(
    num_entities: int,
    edge_index: torch.LongTensor,
    triangles: List[Tuple[int, int, int]],
    tetras: List[Tuple[int, int, int, int]],
) -> Any:
    """build gudhi simplex tree from 1-skeleton edges and 2/3-simplices.

    inserts:
        - all vertices [v] for v in range(num_entities)
        - all edges [u, v] from edge_index
        - all triangles and tetras

    filtration can be set to 0.0 for all simplices. currently testing with 0.0, 1.0, 2.0, 3.0 for vertices, edges, triangles, tetras respectively.
    """
    try:
        import gudhi
    except ImportError:
        logger.warning("gudhi is not installed; simplex_tree will be None")
        return None

    st = gudhi.SimplexTree()

    # 0-simplices: ensure all entities exist, even isolated ones
    for v in range(num_entities):
        st.insert([int(v)], filtration=0.0)

    # 1-simplices: from the 1-skeleton
    for u, v in edge_index.t().tolist():
        st.insert([int(u), int(v)], filtration=1.0)

    # 2-simplices: triangles
    for tri in triangles:
        st.insert(sorted(int(x) for x in tri), filtration=2.0)

    # 3-simplices: tetrahedra
    for tet in tetras:
        st.insert(sorted(int(x) for x in tet), filtration=3.0)

    st.initialize_filtration() #explicitly recommended by documentation after inserting faces etc manually, it's a consistency check and sorting step for other tasks

    logger.info(
        "built simplex tree with %d vertices, %d edges, %d triangles, %d tetrahedra",
        num_entities,
        len(edge_index.t().tolist()),
        len(triangles),
        len(tetras),
    )

    return st


def simplices_from_simplex_tree(
    simplex_tree: Any,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """derive triangle and tetrahedra tensors from a simplex tree.

    treats the simplex tree as the central hub:
    all 3- and 4-vertex simplices in the tree are the triangles / tetrahedra.
    """
    if simplex_tree is None:
        raise ValueError("simplex_tree is None; cannot derive simplices")

    triangles: List[Tuple[int, int, int]] = []
    tetras: List[Tuple[int, int, int, int]] = []

    for simplex, _ in simplex_tree.get_simplices():
        k = len(simplex)
        if k == 3:
            tri = tuple(sorted(int(v) for v in simplex))
            triangles.append(tri)
        elif k == 4:
            tet = tuple(sorted(int(v) for v in simplex))
            tetras.append(tet)

    # sort and deduplicate for deterministic output
    triangles = sorted(set(triangles))
    tetras = sorted(set(tetras))

    if triangles:
        triangles_tensor = torch.tensor(triangles, dtype=torch.long)
    else:
        triangles_tensor = torch.empty((0, 3), dtype=torch.long)

    if tetras:
        tetras_tensor = torch.tensor(tetras, dtype=torch.long)
    else:
        tetras_tensor = torch.empty((0, 4), dtype=torch.long)

    logger.info(
        "derived simplices in tensors from simplex tree: %d triangles, %d tetrahedra",
        triangles_tensor.size(0),
        tetras_tensor.size(0),
    )

    return triangles_tensor, tetras_tensor


def build_bipartite_incidence(
    num_entities: int,
    triangles: torch.LongTensor,
    tetras: torch.LongTensor,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """build entity-triangle, triangle-tetra, and entity-tetra incidence edge_index.

    returns:
        entity_triangle_index: [2, num_entity_triangle_edges]
        triangle_tetra_index: [2, num_triangle_tetra_edges]
        entity_tetra_index: [2, num_entity_tetra_edges]
    """
    num_triangles = triangles.size(0)
    num_tetras = tetras.size(0)

    # ----- entity–triangle incidence -----
    et_edges: List[Tuple[int, int]] = []
    for tri_id in range(num_triangles):
        for v in triangles[tri_id].tolist():
            et_edges.append((v, tri_id))

    if et_edges:
        et_tensor = torch.tensor(et_edges, dtype=torch.long).t().contiguous()
    else:
        et_tensor = torch.empty((2, 0), dtype=torch.long)

    # map triangle (v1, v2, v3) -> triangle_id for fast lookup
    # all this does is just map each triangle from its constituting vertices (unique and sorted to be identifiable) to the triangle id it already has
    # needed because tetra faces are generated as vertex triples
    # this might be improved in efficiency if we did the work during the discovery phase of cliques etc to have a mapping between triangles and tetras

    tri_to_id: Dict[Tuple[int, int, int], int] = {}
    for tri_id in range(num_triangles):
        tri = tuple(sorted(triangles[tri_id].tolist()))
        tri_to_id[tri] = tri_id

    # ----- triangle–tetra incidence -----
    tt_edges: List[Tuple[int, int]] = []
    for tet_id in range(num_tetras):
        vertices = tetras[tet_id].tolist() # extract the verticies that constitute the tetrahedron with id tet_id
        for face in combinations(vertices, 3): # generate all 3-vertex faces of the tetrahedron
            tri = tuple(sorted(face)) # sorting guarantees consistency with how triangles were stored earlier
            tri_id = tri_to_id.get(tri, None) # look up the triangle’s index
            if tri_id is None:
                # this should not happen if simplices were extracted consistently
                logger.debug(
                    "missing triangle %s for tetra %d; skipping that face",
                    tri,
                    tet_id,
                )
                continue
            tt_edges.append((tri_id, tet_id))

    if tt_edges:
        tt_tensor = torch.tensor(tt_edges, dtype=torch.long).t().contiguous()
    else:
        tt_tensor = torch.empty((2, 0), dtype=torch.long)

    # ----- entity–tetra incidence (for downward aggregation) -----
    etet_edges: List[Tuple[int, int]] = []
    for tet_id in range(num_tetras):
        for v in tetras[tet_id].tolist():
            etet_edges.append((v, tet_id))

    if etet_edges:
        etet_tensor = torch.tensor(etet_edges, dtype=torch.long).t().contiguous()
    else:
        etet_tensor = torch.empty((2, 0), dtype=torch.long)

    logger.info(
        "built incidences: |E*tri|=%d, |tri*tet|=%d, |E*tet|=%d",
        et_tensor.size(1),
        tt_tensor.size(1),
        etet_tensor.size(1),
    )

    return et_tensor, tt_tensor, etet_tensor


def build_topology_from_triples(
    num_entities: int,
    triples: torch.LongTensor,
    max_triangles_per_entity: Optional[int] = None,
    max_tetras_per_entity: Optional[int] = None,
) -> TopologyData:
    """end-to-end construction of simplicial views from triples.

    this is the main entry point used by the training pipeline.

    steps:
        1) build 1-skeleton from triples (undirected entity graph)
        2) extract triangles and tetrahedra via clique enumeration (networkx)
        3) build simplex tree from the chosen simplices (central hub)
        4) derive triangle / tetra tensors from simplex tree
        5) build bipartite incidence structures for gnn2/gnn3 + downward pass
    """
    logger.info("starting topology construction")
    edge_index = build_1_skeleton(num_entities=num_entities, triples=triples)

    triangles_list, tetras_list = extract_simplicial_structure(
        num_entities=num_entities,
        edge_index=edge_index,
        max_triangles_per_entity=max_triangles_per_entity,
        max_tetras_per_entity=max_tetras_per_entity,
    )

    simplex_tree = build_simplex_tree(
        num_entities=num_entities,
        edge_index=edge_index,
        triangles=triangles_list,
        tetras=tetras_list,
    )

    if simplex_tree is not None:
        triangles, tetras = simplices_from_simplex_tree(simplex_tree)
    else: # this is incase of implementing a proper optional check for the simplex tree
        # fallback: construct tensors directly from networkx output
        logger.warning("using fallback tensors from networkx output (no simplex tree)")
        triangles = (
            torch.tensor(triangles_list, dtype=torch.long)
            if len(triangles_list) > 0
            else torch.empty((0, 3), dtype=torch.long)
        )
        tetras = (
            torch.tensor(tetras_list, dtype=torch.long)
            if len(tetras_list) > 0
            else torch.empty((0, 4), dtype=torch.long)
        )

    et_index, tt_index, etet_index = build_bipartite_incidence(
        num_entities=num_entities,
        triangles=triangles,
        tetras=tetras,
    )

    topo = TopologyData(
        num_entities=num_entities,
        triangles=triangles,
        tetras=tetras,
        entity_triangle_index=et_index,
        triangle_tetra_index=tt_index,
        entity_tetra_index=etet_index,
        simplex_tree=simplex_tree,
    )

    logger.info(
        "finished topology construction: %d triangles, %d tetrahedra",
        triangles.size(0),
        tetras.size(0),
    )
    return topo
