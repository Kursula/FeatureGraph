import math
import sys
import numpy as np
from scipy.spatial import Delaunay

from .graph import Graph


def get_path_distance(path_keys : list, graph : Graph) -> float:
    total_dist = 0 
    for i in range(len(path_keys) - 1):
        key_a = path_keys[i]
        key_b = path_keys[i + 1]     
        total_dist += graph.vertices[key_a].edges_out[key_b].params['distance']
    return total_dist


def delaunay_edges(graph : Graph, param : str = 'location') -> None:
    """
    Define edges between the vertices using Delaunay triangulation 
    algorithm. 
    
    Triangulation is done using vertex vector parameter. The parameter name 
    is given as input. 
    """
    points = []
    keys = []
    vertices = graph.get_vertices()
    for vertex_key in vertices:
        vect = graph.get_vertex_param(vertex_key, param)
        points.append(vect)
        keys.append(vertex_key)

        # Delete old edges of the vertex
        graph.delete_edge(vertex_key)

    # Calculate triangulation
    points = np.array(points)
    tri = Delaunay(points)
    tripts = tri.simplices
    
    # Add the new edges to the graph
    for row in range(tripts.shape[0]):
        for idx in range(tripts.shape[1]):
            idx_a = tripts[row, idx]
            if idx == tripts.shape[1] - 1:
                idx_b = tripts[row, 0]
            else:
                idx_b = tripts[row, idx + 1]

            key_a = keys[idx_a]
            key_b = keys[idx_b]
            vect_a = graph.get_vertex_param(key_a, param)
            vect_b = graph.get_vertex_param(key_a, param)
            dist = np.linalg.norm(vect_a - vect_b)
            graph.add_edge(key_a, key_b, distance=dist, bidirectional=True)


   
def find_shortest_path(graph : Graph, key_a : any, key_b : any) -> list:
    """
    Algorithm to find the shortest path between two vertices in the graph. 
    This is based on Dijkstra's algorithm that is adapted from various reference
    implementations. 
    
    The algorithm uses edge distances to find the shortest path, so the shortest 
    path might not be the path with smallest number of edges. 
    
    Args: 
        key_a : key for the path starting vertex
        key_b : key for the path ending vertex
        
    Returns: 
        path : list of path keys if the path was found. Returns False if 
        path is not found. 
    """
    shortest_paths = {key_a: (None, 0)}
    current_vertex = key_a
    visited = set()
    vertices = graph.get_vertices()
    
    if (not key_a in vertices) or (not key_b in vertices):
        raise ValueError('Key not found')

    while current_vertex != key_b:
        visited.add(current_vertex)
        destinations = graph.get_edges(current_vertex)['out']
        weight_to_current_vertex = shortest_paths[current_vertex][1]

        for next_vertex in destinations:
            weight = get_path_distance([current_vertex, next_vertex], graph) + weight_to_current_vertex
            if next_vertex not in shortest_paths:
                shortest_paths[next_vertex] = (current_vertex, weight)
            else:
                current_shortest_weight = shortest_paths[next_vertex][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_vertex] = (current_vertex, weight)

        next_destinations = {vertex: shortest_paths[vertex] for vertex in shortest_paths if vertex not in visited}
        if not next_destinations:
            return False
        # next vertex is the destination with the lowest weight
        current_vertex = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []
    while current_vertex is not None:
        path.append(current_vertex)
        next_vertex = shortest_paths[current_vertex][0]
        current_vertex = next_vertex
    # Reverse path
    path = path[::-1]
    return path

