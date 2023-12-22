import machine_searchers
import networkx as nx


similarity = None
graph = None


def set_graph(a_graph):
    global graph
    graph = a_graph


def set_similarity(sim):
    global similarity
    similarity = sim


def apply_machine_both(row, semantic_similarity = similarity, decoded_wikispeedia: nx.Graph = graph) -> list:
    # print(semantic_similarity)
    # print(decoded_wikispeedia)
    # print(row)
    if decoded_wikispeedia is None:
        return

    source = row['first_article']
    target = row['last_article']

    lib_path_1, lib_explore_1 = machine_searchers.modded_astar_path(decoded_wikispeedia, source, target, heuristic=semantic_similarity)
    lib_path_2, lib_explore_2 = machine_searchers.only_depth_first_astar_path(decoded_wikispeedia, source, target, heuristic=semantic_similarity)

    return [source, target, len(lib_explore_1)-1, lib_path_1, lib_explore_1, len(lib_explore_2)-1, lib_path_2, lib_explore_2]


def apply_machine_first(row, semantic_similarity = similarity, decoded_wikispeedia: nx.Graph = graph) -> list:
    if decoded_wikispeedia is None:
        return

    source = row['first_article']
    target = row['last_article']

    lib_path_1, lib_explore_1 = machine_searchers.modded_astar_path(decoded_wikispeedia, source, target, heuristic=semantic_similarity)
    #lib_path_2, lib_explore_2 = machine_searchers.only_depth_first_astar_path(decoded_wikispeedia, source, target, heuristic=semantic_similarity)

    return [source, target, len(lib_explore_1)-1, lib_path_1, lib_explore_1]

