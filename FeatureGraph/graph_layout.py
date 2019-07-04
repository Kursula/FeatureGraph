import math
import sys
import numpy as np

from .graph import Graph


class ProgressPrint: 
    def __init__(self, max_step, n_steps=20):
        """
        Internal function to initialize progress print. 
        """
        self.step_divider = max_step / n_steps
        self.prev_step = None
        print(' ') # new line
        
        
    def print_update(self, iteration, step, stress_value): 
        """
        Progress print used by the arrange_vertices function.
        """
        print_step = int(step / self.step_divider)
        if print_step == self.prev_step:
            return 
        self.prev_step = print_step

        sys.stdout.write(f'\r' + 'Iteration {}'.format(iteration) +
                         ' [' + '=' * (print_step) + '>' + 
                         '.' * (20 - print_step) + '] ' + 
                         'stress = {:.2f}'.format(stress_value) + '   ')
        

class LearningRateGen:

    def __init__(self, lr_start=1, lr_decay=1, lr_base=0.001):
        self.lr_start = lr_start
        self.lr_decay = lr_decay
        self.lr_base = lr_base

    def get_lr(self, iteration): 
        lr = self.lr_start * self.lr_decay ** iteration
        if lr < self.lr_base:
            lr = self.lr_base
        return lr



def __coord_update(graph, key_a, key_b, dist_target, lr):
    """
    Internal function to update vertex coordinates in the layout
    optimization loop. 
    """

    # Get current vertex coordinates
    cp_a = np.array(graph.get_vertex_param(key_a, 'location'))
    cp_b = np.array(graph.get_vertex_param(key_b, 'location'))
    cp_diff = np.linalg.norm(cp_a - cp_b)
    if cp_diff == 0: 
        return 0 # This is necessary to avoid div by zero issues.

    # Calculate coordinate update step size
    if dist_target != 0: 
        upd_scaling = (dist_target ** -2) * lr
    else: 
        upd_scaling = 1 
    if upd_scaling > 1: 
        upd_scaling = 1                    

    dist_error = (cp_diff - dist_target) / (2 * cp_diff)    
    dist_upd = (cp_a - cp_b) * dist_error * upd_scaling

    # Store new coordinates
    graph.set_vertex_param(key_a, location=cp_a - dist_upd)
    graph.set_vertex_param(key_b, location=cp_b + dist_upd)

    # Calculate stress 
    stress = math.fabs(cp_diff - dist_target)
    return stress


def create_adjacency_matrix(graph : Graph) -> np.ndarray:
    dim = graph.vertex_count
    adj = np.full((dim, dim), np.inf)
    vertices = graph.get_vertices()

    for col_idx, col_vt_key in enumerate(vertices):
        col_vt_edges = graph.get_edges(col_vt_key)['out']

        for row_idx, row_vt_key in enumerate(vertices):
            if row_vt_key in col_vt_edges:
                dist = graph.get_edge_param(col_vt_key, row_vt_key, 'distance')
                adj[col_idx, row_idx] = dist
    
            elif col_idx == row_idx:   
                adj[col_idx, row_idx] = 0
    
    return adj, vertices

                                                         
def floyd_warshall(adjacency_matrix : np.ndarray) -> np.ndarray: 
    """
    Calculate shortest path distance between all vertices in graph. 
    """
    dist = adjacency_matrix.copy()
    dim = adjacency_matrix.shape[0]
    
    for k in range(dim): 
        for i in range(dim): 
            for j in range(dim): 
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])

    print(dist)
    return dist


def __edge_distance_lut(graph : Graph) -> (np.ndarray, list):
    """
    Calculate look-up table for edge distances between all vertices. 
    """
    # Calculate adjacency matrix 
    adj_matr, vertex_keys = create_adjacency_matrix(graph)
    
    # Calculate shortest path distance between all vertices 
    dists = floyd_warshall(adj_matr)

    return dists, vertex_keys



def __vertex_vector_distance_lut(graph : Graph) -> (np.ndarray, list):
    """
    Calculate look-up table for vertex vector distances
    """

    # Create temporary lists of vertex list indices
    n_vertex = graph.vertex_count
    vertex_keys = graph.get_vertices()
    dist_arr = np.zeros((n_vertex, n_vertex), dtype=np.float32)

    for i, key_a in enumerate(vertex_keys):
        vect_a = graph.get_vertex_param(key_a, 'feature_vector')

        for j in range(i + 1, n_vertex):
            key_b = vertex_keys[j]
            vect_b = graph.get_vertex_param(key_b, 'feature_vector')
            dist = np.linalg.norm(vect_a - vect_b)
            dist_arr[i, j] = dist
            dist_arr[j, i] = dist

    return dist_arr, vertex_keys


def __reset_locations(graph : Graph, dim : int = 2) -> None:
    """
    Initialize all vertex location parameters to random values. 
    """
    vertex_keys = graph.get_vertices()
    for key in vertex_keys:
        graph.set_vertex_param(key, location=np.random.rand(dim))


def edge_distance_mapping(graph : Graph, 
                          iterations : int, 
                          lrgen : LearningRateGen, 
                          verbose : bool = True, 
                          reset_locations : bool = True):        
    """
    Stochastic Gradient Descent algorithm for performing graph vertex laoyout 
    optimization using the path distances as target distance in the layout. 
    The algorihm is adapted from the paper https://arxiv.org/pdf/1710.04626.pdf
    
    Args: 
        graph : The graph to arrange
        iterations : number of iteration rounds
        lrgen : learning rate function that takes iteration round as input
        verbose : boolean, set True to print progress status information
        
    Returns: 
        Vertex location stress value list that contains one summary stress 
        value per iteration. 
    """

    # Create temporary lists of vertex list indices
    n_vertex = graph.vertex_count
    vertex_idx_list_a = np.arange(n_vertex)
    vertex_idx_list_b = np.arange(n_vertex)
    stress_list = []

    # Calculate distance look-up table
    dist_arr, keys = __edge_distance_lut(graph)

    if reset_locations:
        __reset_locations(graph)

    # Main iteration loop
    for iter_round in range(iterations):
        stress = 0
        lr = lrgen.get_lr(iter_round)
        
        if verbose:
            progress_print = ProgressPrint(n_vertex)
            a_loop = 0
                
        np.random.shuffle(vertex_idx_list_a)
        for idx_a in vertex_idx_list_a:

            np.random.shuffle(vertex_idx_list_b)
            for idx_b in vertex_idx_list_b:
                if idx_a == idx_b:
                    continue

                # Get path distance from vertex a to b.
                # Value -1 means there is no path. 
                dist_target = dist_arr[idx_a, idx_b]
                if dist_target == -1:
                    continue
                
                # Update the locations and get stress for the patg
                key_a = keys[idx_a]
                key_b = keys[idx_b]
                edge_stress = __coord_update(graph, key_a, key_b, dist_target, lr)
                stress += edge_stress
            
            # Progress monitoring
            if verbose: 
                a_loop += 1
                progress_print.print_update(iter_round, a_loop, stress)
            
        stress_list.append(stress)
    return stress_list


def vertex_vector_mapping(graph : Graph, 
                          iterations : int, 
                          lrgen : LearningRateGen, 
                          verbose : bool = True, 
                          reset_locations : bool = True):  
    """
    Stochastic Gradient Descent algorithm for performing Sammon mapping of the 
    graph vertices for 2D plotting. Vertex feature vectors are used to guide the 
    layout optimization. Graph edges are not used at all in this method, 

    The algorihm is adapted from the paper https://arxiv.org/pdf/1710.04626.pdf
    
    Args: 
        graph : The graph to arrange
        iterations : number of iteration rounds
        lrgen : learning rate function that takes iteration round as input
        verbose : boolean, set True to print progress status information
        
    Returns: 
        Vertex location stress value list that contains one summary stress 
        value per iteration. 
    """

    # Create temporary lists of vertex list indices
    n_vertex = graph.vertex_count
    vertex_idx_list_a = np.arange(n_vertex)
    vertex_idx_list_b = np.arange(n_vertex)
    stress_list = []

    dist_arr, keys = __vertex_vector_distance_lut(graph)

    if reset_locations:
        __reset_locations(graph)

    # Main iteration loop
    for iter_round in range(iterations):
        stress = 0
        lr = lrgen.get_lr(iter_round)
        if verbose:
            progress_print = ProgressPrint(n_vertex)
            a_loop = 0
                
        # Loop through all vertices in random order
        np.random.shuffle(vertex_idx_list_a)
        for idx_a in vertex_idx_list_a:

            # Loop through all vertices except the idx_a
            np.random.shuffle(vertex_idx_list_b)
            for idx_b in vertex_idx_list_b:
                if idx_a == idx_b:
                    continue

                # Get target distance from vertex idx_a to vertex idx_b. This ignores all
                # edges i.e. the distance is based only on the feature vectors. 
                dist_target = dist_arr[idx_a, idx_b]

                # Calculate the actual distance between the vertices in their current 
                # locations. The graph distance method is used for this to ensure that 
                # the distance calculation is same for all distances. 
                key_a = keys[idx_a]
                key_b = keys[idx_b]
                res_stress = __coord_update(graph, key_a, key_b, dist_target, lr)
                stress += res_stress
            
            # Progress monitoring
            if verbose: 
                a_loop += 1
                progress_print.print_update(iter_round, a_loop, stress)
            
        stress_list.append(stress)
    return stress_list

