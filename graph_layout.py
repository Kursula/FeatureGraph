import math
import sys
import numpy as np


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


class GraphLayoutMixin:

    def _coord_update(self, vertex_a, vertex_b, dist_target, lr):
        """
        Internal function to update vertex coordinates in the layout
        optimization loop. 
        """

        cp_a = np.array([vertex_a.loc_x, vertex_a.loc_y])
        cp_b = np.array([vertex_b.loc_x, vertex_b.loc_y])
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

        vertex_a.loc_x -= dist_upd[0]   
        vertex_a.loc_y -= dist_upd[1]   
        vertex_b.loc_x += dist_upd[0] 
        vertex_b.loc_y += dist_upd[1] 

        # Calculate stress 
        stress = math.fabs(cp_diff - dist_target)
        return stress


    def _distance_lut(self, method):

        # Create temporary lists of vertex list indices
        n_vertex = len(self.vertices)
        keys = list(self.vertices.keys())
        dist_arr = np.zeros((n_vertex, n_vertex), dtype=np.float32)

        # Calculate distance look-up table
        if method == 'shortest_path':
            """
            This method calculates the edge path distances between vertices. 
            Can be slow with large graphs. 
            """
            for i in range(n_vertex):
                key_a = keys[i]
                for j in range(n_vertex):
                    key_b = keys[j]
                    path = self.find_shortest_path(key_a, key_b)
                    if path == False: 
                        dist = -1 
                    else:
                        dist = self.get_path_dist(path)
                    dist_arr[i, j] = dist

        elif method == 'feature_distance':
            """
            This method does not care about graph edges. 
            Distance is calculated as Euclidean distance between 
            the vertex feature vectors. 
            """
            for i in range(n_vertex):
                vect_a = self.vertices[keys[i]].params['feature_vector']
                for j in range(i + 1, n_vertex):
                    vect_b = self.vertices[keys[j]].params['feature_vector']
                    dist = np.linalg.norm(vect_a - vect_b)
                    dist_arr[i, j] = dist
                    dist_arr[j, i] = dist

        else:
            raise ValueError('Incorrect distance method')

        return dist_arr, keys


    def _reset_locations(self):
        for vertex in self.vertices.values():
            vertex.loc_x = np.random.rand()
            vertex.loc_y = np.random.rand()


    def distance_mapping(self, iterations, lrgen, verbose=True, reset_locations=True):        
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
        n_vertex = len(self.vertices)
        vertex_idx_list_a = np.arange(n_vertex)
        vertex_idx_list_b = np.arange(n_vertex)
        stress_list = []


        # Calculate distance look-up table
        dist_arr, keys = self._distance_lut(method='shortest_path')

        if reset_locations:
            self._reset_locations()

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
                    vertex_a = self.vertices[keys[idx_a]]
                    vertex_b = self.vertices[keys[idx_b]]
                    edge_stress = self._coord_update(vertex_a, vertex_b, dist_target, lr)
                    stress += edge_stress
                
                # Progress monitoring
                if verbose: 
                    a_loop += 1
                    progress_print.print_update(iter_round, a_loop, stress)
                
            stress_list.append(stress)
        return stress_list


    def sammon_mapping(self, iterations, lrgen, verbose=True, reset_locations=True):        
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
        n_vertex = len(self.vertices)
        vertex_idx_list_a = np.arange(n_vertex)
        vertex_idx_list_b = np.arange(n_vertex)
        stress_list = []

        dist_arr, keys = self._distance_lut(method='feature_distance')

        if reset_locations:
            self._reset_locations()

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
                    vertex_a = self.vertices[keys[idx_a]]
                    vertex_b = self.vertices[keys[idx_b]]
                    res_stress = self._coord_update(vertex_a, vertex_b, dist_target, lr)
                    stress += res_stress
                
                # Progress monitoring
                if verbose: 
                    a_loop += 1
                    progress_print.print_update(iter_round, a_loop, stress)
                
            stress_list.append(stress)
        return stress_list

    