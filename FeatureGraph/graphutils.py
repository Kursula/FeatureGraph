import math
import sys
import numpy as np
from scipy.spatial import Delaunay


class GraphUtilsMixin:
    
    def get_path_distance(self, key_list):
        total_dist = 0 
        for i in range(len(key_list) - 1):
            key_a = key_list[i]
            key_b = key_list[i + 1]     
            total_dist += self.vertices[key_a].edges_out[key_b].params['distance']
        return total_dist
    
    
    def get_edges(self, key):
        if not key in self.vertices:
            raise ValueError('Key not found')
        
        edges_out = list(self.vertices[key].edges_out.keys())
        edges_in = list(self.vertices[key].edges_in.keys())
        return {'in' : edges_in, 'out' : edges_out}

    def get_vertices(self):
        keys = list(self.vertices.keys())
        return keys


    def distance(self, key_a, key_b, param='feature_vector'):
        """
        Calculates euclidean distance between vertex feature vectors.
        """
        vect_a = self.vertices[key_a].params[param]
        vect_b = self.vertices[key_b].params[param]
        dist = np.linalg.norm(vect_a - vect_b)
        return dist


    def delaunay_edges(self, param='location'):
        """
        Define edges between the vertices using Delaunay triangulation 
        algorithm. 
        
        Triangulation is done using vertex vector parameters. The parameter name 
        is given as input. 
        """
        points = []
        keys = []
        for vertex in self.vertices.values():
            points.append(vertex.params[param])
            keys.append(vertex.key)

            # Delete old edges of the vertex
            self.delete_edge(vertex.key)

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
                dist = self.distance(key_a, key_b, param=param)
                self.add_edge(key_a, key_b, distance=dist, bidirectional=True)


       
    def find_shortest_path(self, key_a, key_b):
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
        
        if (not key_a in self.vertices) or (not key_b in self.vertices):
            raise ValueError('Key not found')

        while current_vertex != key_b:
            visited.add(current_vertex)
            destinations = self.get_edges(current_vertex)['out']
            weight_to_current_vertex = shortest_paths[current_vertex][1]

            for next_vertex in destinations:
                weight = self.get_path_distance([current_vertex, next_vertex]) + weight_to_current_vertex
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

