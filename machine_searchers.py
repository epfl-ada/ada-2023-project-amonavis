from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
import pandas as pd

class LandmarkSearch:
    def __init__(self, graph: nx.DiGraph, landmark_num: int = 50):
        # Default value should be a function of the size of the graph...
        self.landmark_num = landmark_num

        self.landmark_node_list = None

        # Empty dictionaries to store info
        self.shortest_paths_to_node = {}
        self.shortest_paths_from_node = {}

        self.fro_df = None
        self.to_df = None

        self.get_landmark_info(graph, landmark_num)

    def get_landmark_info(self, graph: nx.DiGraph, landmark_num: int):
        temp = sorted(graph.degree, key=lambda x: x[1], reverse=True)
        temp = [elem[0] for elem in temp]
        self.landmark_node_list = temp[:landmark_num]

        for elem in self.landmark_node_list:
            self.shortest_paths_to_node[elem] = nx.single_target_shortest_path(graph, elem)
            self.shortest_paths_from_node[elem] = nx.single_source_shortest_path(graph, elem)

        # Transforming the previous elements into a dict of lengths, because it's important
        # But it's a dict of dicts!
        paths_to_lengths = {}
        paths_fro_lengths = {}

        max_length = len(graph.nodes)

        for elem in graph.nodes:
            paths_fro_lengths[elem] = {}
            paths_to_lengths[elem] = {}
            for landmark in self.shortest_paths_from_node.keys():
                # This extra code is to check if the key exists or not in the dictionaries

                # And fro and to are swapped, but that's because the dicts we save the info to
                # are as well.
                # So this ends up making sense
                if elem in self.shortest_paths_from_node[landmark]:
                    paths_to_lengths[elem][landmark] = len(self.shortest_paths_from_node[landmark][elem])
                else:
                    paths_to_lengths[elem][landmark] = max_length

                if elem in self.shortest_paths_to_node[landmark]:
                    paths_fro_lengths[elem][landmark] = len(self.shortest_paths_to_node[landmark][elem])
                else:
                    paths_fro_lengths[elem][landmark] = max_length

        # The easy way of distinguishing the two dfs is as follows:
        # Get a loc[a, b]
        # fro_df will describe distance from b to a
        # to_df describes distance from a to b
        self.fro_df = pd.DataFrame(paths_fro_lengths)
        self.to_df = pd.DataFrame(paths_to_lengths)

    def find_shortest_path(self, source, target):
        # For this, I sum up the two and fro somehow, and find the values!
        temp_fro = self.fro_df.loc[:, source]
        temp_to = self.to_df.loc[:, target]

        distances = temp_to + temp_fro
        distances.sort_values(inplace=True)

        landmark = distances.index[0]

        # The landmark is the middle point, this tells us the best one
        start_path = self.shortest_paths_to_node[landmark][source][:-1]
        end_path = self.shortest_paths_from_node[landmark][target]

        final_path = start_path + end_path

        return final_path

def modded_astar_path(G: nx.Graph, source: str, target: str, heuristic=None, weight="weight"):
    """Returns a list of nodes in a shortest path between source and target
    using the A* ("A-star") algorithm.

    There may be more than one shortest path.  This returns only one.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    heuristic : function
       A function to evaluate the estimate of the distance
       from the a node to the target.  The function takes
       two nodes arguments and must return a number.
       If the heuristic is inadmissible (if it might
       overestimate the cost of reaching the goal from a node),
       the result may not be a shortest path.
       The algorithm does not support updating heuristic
       values for the same node due to caching the first
       heuristic calculation per node.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.
       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number or None to indicate a hidden edge.

    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> print(nx.astar_path(G, 0, 4))
    [0, 1, 2, 3, 4]
    >>> G = nx.grid_graph(dim=[3, 3])  # nodes are two-tuples (x,y)
    >>> nx.set_edge_attributes(G, {e: e[1][0] * 2 for e in G.edges()}, "cost")
    >>> def dist(a, b):
    ...     (x1, y1) = a
    ...     (x2, y2) = b
    ...     return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    >>> print(nx.astar_path(G, (0, 0), (2, 2), heuristic=dist, weight="cost"))
    [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    See Also
    --------
    shortest_path, dijkstra_path

    """
    if source not in G or target not in G:
        msg = f"Either source {source} or target {target} is not in G"
        raise nx.NodeNotFound(msg)

    if heuristic is None:
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        def heuristic(u, v):
            return 0

    push = heappush
    pop = heappop
    weight = _weight_function(G, weight)

    g_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)

    # The queue stores priority, node, cost to reach, and parent.
    # Uses Python heapq to keep in priority order.
    # Add a counter to the queue to prevent the underlying heap from
    # attempting to compare the nodes themselves. The hash breaks ties in the
    # priority and is guaranteed unique for all nodes in the graph.
    c = count()
    queue = [(0, next(c), source, 0, None)]

    # Maps enqueued nodes to distance of discovered paths and the
    # computed heuristics to target. We avoid computing the heuristics
    # more than once and inserting the node into the queue too many times.
    enqueued = {}
    # Maps explored nodes to parent closest to the source.
    explored = {}

    while queue:
        # Pop the smallest item from queue.
        _, __, curnode, dist, parent = pop(queue)

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path, explored

        if curnode in explored:
            # Do not override the parent of starting node
            if explored[curnode] is None:
                continue

            # Skip bad paths that were enqueued before finding a better one
            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue

        explored[curnode] = parent

        for neighbor, w in g_succ[curnode].items():
            # This is the real only place where the code was changed, as I added a check for the neighbor being adjacent
            if neighbor == target:
                explored[neighbor] = curnode
                path = [neighbor]
                node = curnode
                while node is not None:
                    path.append(node)
                    node = explored[node]
                path.reverse()
                return path, explored

            cost = weight(curnode, neighbor, w)
            if cost is None:
                continue
            ncost = dist + cost
            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]
                # if qcost <= ncost, a less costly path from the
                # neighbor to the source was already determined.
                # Therefore, we won't attempt to push this neighbor
                # to the queue
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, target)
            enqueued[neighbor] = ncost, h
            push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

    raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")

def only_depth_first_astar_path(G: nx.Graph, source: str, target: str, heuristic=None):
    """Returns a list of nodes in a shortest path between source and target
    using the A* ("A-star") algorithm.

    There may be more than one shortest path.  This returns only one.

    Only goes depth first, if it finds no nodes it gives up moving onwards

    """
    if source not in G or target not in G:
        msg = f"Either source {source} or target {target} is not in G"
        raise nx.NodeNotFound(msg)

    if heuristic is None:
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        def heuristic(u, v):
            return 0

    g_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)

    # Maps explored nodes to parent closest to the source.
    explored = {}

    curnode = source
    parent = None

    while curnode is not None:

        explored[curnode] = parent

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path, explored

        # Here I explore all the nodes, find the cost and decide only the smallest one
        adjacent_nodes_and_weight = []

        for neighbor, w in g_succ[curnode].items():
            # This is the real only place where the code was changed, as I added a check for the neighbor being adjacent
            if neighbor == target:
                explored[neighbor] = curnode
                path = [neighbor]
                node = curnode
                while node is not None:
                    path.append(node)
                    node = explored[node]
                path.reverse()
                return path, explored

            h = heuristic(neighbor, target)

            adjacent_nodes_and_weight.append((neighbor, h))

        adjacent_nodes_and_weight.sort(key=lambda x: x[1])

        # Now, pick the lowest value that hasn't been explored yet
        # It does mean we could get in a loop, but that's okay!
        parent = curnode
        curnode = None
        #print(parent)

        for node, val in adjacent_nodes_and_weight:
            if not node in explored:
                curnode = node
                break

    print(f"Node {target} not reachable from {source} in depth first version")
    return [], explored