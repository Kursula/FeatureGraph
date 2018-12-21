import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import Delaunay



class GraphUtils(object):
    """
    Collection of graph utilities
    """
    

    def delaunay_edges(self, graph, method='location'):
        """
        Define edges between the vertices using Delaunay triangulation 
        algorithm. 
        
        Edges can be defined based on the vertex (x, y) location, or alternatively
        based on the feature vectors. When using the feature vectors, the algorithm 
        typically works best with 8 or less dimensions. 
        
        Args:
            method : 'location' or 'feature' as described above.
            
        Returns:
            Nothing
        """
        
        # Create list of coordinates for the triangulation.
        points = []
        keys = []
        for vertex in graph.vertices.values():
            if method == 'location':
                points.append([vertex.loc_x, vertex.loc_y])
            elif method == 'feature':
                points.append(vertex.feature_vector)
            else: 
                raise ValueError('Unknown method')

            keys.append(vertex.key)
            # Reset old edges of the vertex
            vertex.edges = {}

        # Calculate triangulation
        points = np.array(points)
        tri = Delaunay(points)
        tripts = tri.simplices
        
        # Add new edges to the graph
        for row in range(tripts.shape[0]):
            for idx in range(tripts.shape[1]):
                idx_a = tripts[row, idx]
                if idx == tripts.shape[1] - 1:
                    idx_b = tripts[row, 0]
                else:
                    idx_b = tripts[row, idx + 1]

                graph.add_edge(keys[idx_a], keys[idx_b])


    
    def init_progress_print(self, max_step):
        """
        Internal function to initialize progress print. 
        """
        self.step_divider = max_step / 20.0
        print(' ') # new line
        
        
    def progress_print(self, iteration, step, stress_value): 
        """
        Progress print used by the arrange_vertices function.
        """
        print_step = int(step / self.step_divider)
        sys.stdout.write(f'\r' + 'Iteration {}'.format(iteration) +
                         ' [' + '=' * (print_step) + '>' + 
                         '.' * (20 - print_step) + '] ' + 
                         'stress = {:.2f}'.format(stress_value) + '   ')
        

    def arrange_vertices(self, graph, iterations, lr=1, verbose=True):        
        """
        Stochastic Gradient Descent algorithm for arranging the 
        graph vertices for 2D plotting. 
        The algorihm is adapted from the paper https://arxiv.org/pdf/1710.04626.pdf
        
        Args: 
            graph : The graph to arrange
            iterations : number of iteration rounds
            lr : learning rage (a.k.a. update step size)
            verbose : boolean, set True to print progress status information
            
        Returns: 
            Vertex location stress value list that contains one summary stress 
            value per iteration. 
        """

        # Create temporary lists of vertex list indices
        n_vertex = len(graph.vertices)
        vertex_idx_list_a = np.arange(n_vertex)
        vertex_idx_list_b = np.arange(n_vertex)
        keys = list(graph.vertices.keys())

        # Calculate distance look-up table
        dist_arr = np.zeros((n_vertex, n_vertex), dtype=np.float32)
        for i in range(n_vertex):
            vect_a = graph.vertices[keys[i]].feature_vector
            j = i + 1
            for j in range(i +1, n_vertex):
                vect_b = graph.vertices[keys[j]].feature_vector
                dist = graph.distance(vect_a, vect_b)
                dist_arr[i, j] = dist
                dist_arr[j, i] = dist

        stress_list = []

        # Main iteration loop
        for iter_round in range(iterations):
            stress = 0

            # Init progress monitor for this iteration
            if verbose:
                self.init_progress_print(n_vertex)
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
                    vertex_a = graph.vertices[keys[idx_a]]
                    vertex_b = graph.vertices[keys[idx_b]]

                    cp_a = np.array([vertex_a.loc_x, vertex_a.loc_y])
                    cp_b = np.array([vertex_b.loc_x, vertex_b.loc_y])

                    cp_diff = graph.distance(cp_a, cp_b)
                    if cp_diff == 0: 
                        continue

                    dist_error = (cp_diff - dist_target) / (2 * cp_diff)    
    
                    # Calculate coordinate update step size
                    if dist_target != 0: 
                        upd_scaling = (dist_target ** -2) * lr
                    else: 
                        upd_scaling = 1 
                    if upd_scaling > 1: 
                        upd_scaling = 1                    

                    dist_upd = (cp_a - cp_b) * dist_error * upd_scaling

                    # Update the vertex coordinatex
                    vertex_a.loc_x -= dist_upd[0]   
                    vertex_a.loc_y -= dist_upd[1]   
                    vertex_b.loc_x += dist_upd[0] 
                    vertex_b.loc_y += dist_upd[1] 

                    # Calculate stress 
                    stress += math.fabs(cp_diff - dist_target)
                
                # Progress monitoring
                if verbose: 
                    a_loop += 1
                    self.progress_print(iter_round, a_loop, stress)
                
            stress_list.append(stress)
        return stress_list
    
        
    def find_shortest_path(self, graph, key_a, key_b):
        """
        Algorithm to find the shortest path between two vertices in the graph. 
        This is based on Dijkstra's algorithm that is adapted from various reference
        implementations. 
        
        The algorithm uses edge distances (feature vector distances) to find the 
        shortest path, so the shortest path might not be the path with smallest 
        number of edges. 
        
        Args: 
            graph : the graph to be used. 
            key_a : key for the path starting vertex
            key_b : key for the path ending vertex
            
        Returns: 
            path : list of path keys if the path was found. Returns None if 
            path is not found. 
        """
        shortest_paths = {key_a: (None, 0)}
        current_vertex = key_a
        visited = set()
        
        if (not key_a in graph.vertices) or (not key_b in graph.vertices):
            raise ValueError('Key not found')

        while current_vertex != key_b:
            visited.add(current_vertex)
            destinations = graph.get_edges(current_vertex)
            weight_to_current_vertex = shortest_paths[current_vertex][1]

            for next_vertex in destinations:
                weight = graph.get_path_dist([current_vertex, next_vertex]) + weight_to_current_vertex
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


class GraphPlot: 
    
    def __init__(self, img_x=2000, img_y=2000, 
                 plot_size=(13, 13),
                 pad_ratio=0.1,
                 background_color=[255, 255, 255]):
        
        self.img_x = img_x
        self.img_y = img_y
        self.plot_size = plot_size
        self.pad_ratio = pad_ratio
        self.background_color = background_color
        
        # initialize the plot canvas
        self.canvas = np.zeros((img_y, img_x, 3), dtype=np.uint8) 
        self.canvas[:, :] = background_color

        
    def set_graph(self, graph):
        """
        Load graph to the plot object. 
        """
        self.graph = graph
        self._get_min_max_coords()
        self._add_padding()
        
        
    def plot_title(self, title_str, loc=(100, 100), scale=2, thickness=2, 
                   color=[0, 0, 0]):
        """
        Plot title string on the canvas.
        """
        cv2.putText(self.canvas, title_str, loc, 
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale=scale, thickness=thickness, color=color)
        
        
    def plot_edges(self, color=[0, 0, 0], width=1):
        """
        Plot all edges in the graph. 
        """
        for vertex_a in self.graph.vertices.values(): 
            for edge_key in vertex_a.edges.keys():
                vertex_b = self.graph.vertices[edge_key]
                self._plot_edge(vertex_a, vertex_b, color, width)
                
                
    def _plot_edge(self, vertex_a, vertex_b, color, width):
        vtx_a_x = self._scale_x(vertex_a.loc_x)
        vtx_a_y = self._scale_y(vertex_a.loc_y)
        
        vtx_b_x = self._scale_x(vertex_b.loc_x)
        vtx_b_y = self._scale_y(vertex_b.loc_y)

        cv2.line(self.canvas, 
                 (vtx_a_x, vtx_a_y), (vtx_b_x, vtx_b_y), 
                 thickness=width, 
                 lineType=cv2.LINE_AA, 
                 color=color)
                    
                    
    def plot_vertices(self, plot_thumbnails=False,
                      radius=10, edge_color=[0, 0, 0],
                      edge_width=1):
        """
        Plot all vertices of the graph
        """
        
        for vertex in self.graph.vertices.values(): 
            self._plot_vertex(vertex, edge_color, edge_width, 
                              plot_thumbnails, radius)


    def _plot_vertex(self, vertex, edge_color, edge_width, 
                     plot_thumbnails, radius):
        
        cp_x = self._scale_x(vertex.loc_x)
        cp_y = self._scale_y(vertex.loc_y)

        if plot_thumbnails: 
            self._plot_thumbnail(vertex.image, cp_x, cp_y, radius)
        else: # draw solid circle
            cv2.circle(self.canvas, (cp_x, cp_y), radius=radius, 
                       thickness=-1, color=vertex.color)
            
        # Draw  border around the thumbnail or colored vertex
        cv2.circle(self.canvas, (cp_x, cp_y), radius=radius, 
                   lineType=cv2.LINE_AA, 
                   thickness=edge_width, color=edge_color)

            
    def plot_path(self, path_list, plot_thumbnails=False,
                  radius=30, edge_color=[255, 0, 0],
                  edge_width=2):
        
        """
        Highlight the path in the graph plot. 
        """
        
        # Plot edges
        for i in range(len(path_list) - 1): 
            key_a = path_list[i]
            vertex_a = self.graph.vertices[key_a]
            key_b = path_list[i + 1]
            vertex_b = self.graph.vertices[key_b]
            self._plot_edge(vertex_a, vertex_b, edge_color, edge_width)
            
        # Plot vertices
        for key in path_list: 
            vertex = self.graph.vertices[key]
            self._plot_vertex(vertex, edge_color, edge_width, 
                  plot_thumbnails, radius)

            
    def _scale_x(self, x):
        x_step = (self.x_max - self.x_min) / self.img_x
        x_index = int((x - self.x_min) / x_step)
        return x_index
    
    
    def _scale_y(self, y):
        y_step = (self.y_max - self.y_min) / self.img_y
        y_index = int((y - self.y_min) / y_step)
        return y_index
        
        
    def _add_padding(self):
        """
        Add padding to the coordinate ranges. This effectively produces
        empty margins to the final image. 
        """
        self.x_min -= self.pad_ratio * (self.x_max - self.x_min)
        self.x_max += self.pad_ratio * (self.x_max - self.x_min)
        self.y_min -= self.pad_ratio * (self.y_max - self.y_min)
        self.y_max += self.pad_ratio * (self.y_max - self.y_min)

        
    def _get_min_max_coords(self):
        """
        Helper function to get graph vertex location min max coordinates.
        """
        some_key = list(self.graph.vertices.keys())[0]
        self.x_min = self.graph.vertices[some_key].loc_x
        self.x_max = self.graph.vertices[some_key].loc_x
        self.y_min = self.graph.vertices[some_key].loc_y
        self.y_max = self.graph.vertices[some_key].loc_y
        
        for vertex in self.graph.vertices.values():
            if self.x_min > vertex.loc_x:
                self.x_min = vertex.loc_x
            if self.x_max < vertex.loc_x:
                self.x_max = vertex.loc_x
            if self.y_min > vertex.loc_y:
                self.y_min = vertex.loc_y                
            if self.y_max < vertex.loc_y:
                self.y_max = vertex.loc_y 
        

    def _plot_thumbnail(self, thumbnail, loc_x, loc_y, radius):
        """
        Plot vertex thumbnail on the canvas. 
        """
        
        # Check that we are not plotting outside of the canvas
        if (loc_x - radius < 0) or\
           (loc_y - radius < 0):
            return False
        if (loc_x + radius >= self.canvas.shape[1]) or\
           (loc_y + radius >= self.canvas.shape[0]):
            return False
        
        # Create circular mask for the thumbnail plot
        tn_dim = radius * 2
        mask = np.zeros((tn_dim, tn_dim, 3), dtype=np.uint8)
        mask = cv2.circle(mask, (radius, radius), radius=radius, thickness=-1, color=[1, 1, 1])
        
        # Scale and mask the thumbnail image
        tn = cv2.resize(thumbnail.copy(), dsize=(tn_dim, tn_dim))
        tn *= mask
        
        # Erase the thumbnail area from the canvas
        mask = ((mask == 0) * 1).astype(np.uint8)
        self.canvas[loc_y - radius : loc_y - radius + tn_dim, 
                    loc_x - radius : loc_x - radius + tn_dim] *= mask
        
        # Copy thumbnail to the canvas
        self.canvas[loc_y - radius : loc_y - radius + tn_dim, 
                    loc_x - radius : loc_x - radius + tn_dim] += tn 

        return True

    def show(self):
        plt.figure(figsize=self.plot_size)
        plt.imshow(self.canvas, interpolation='sinc')
        plt.axis('off')
        plt.show()

    def save(self, filename):
        cv2.imwrite(filename, cv2.cvtColor(self.canvas, cv2.COLOR_RGB2BGR))



