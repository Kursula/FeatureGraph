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
    

    def define_edges(self, graph, method='location'):
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
        for vertex in graph.vertices:
            if method == 'location':
                points.append([vertex.loc_x, vertex.loc_y])
            elif method == 'feature':
                points.append(vertex.feature_vector)
            else: 
                raise ValueError('Unknown method')

            # Reset old edges
            vertex.edges = []
            vertex.distances = []

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

                key_a = graph.vertices[idx_a].key
                key_b = graph.vertices[idx_b].key
                graph.add_edge(key_a, key_b)


    
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

        # Calculate distance look-up table
        dist_arr = np.zeros((n_vertex, n_vertex), dtype=np.float32)
        for i in range(n_vertex):
            vect_a = graph.vertices[i].feature_vector
            j = i + 1
            while j < n_vertex:
                vect_b = graph.vertices[j].feature_vector
                dist = graph.distance(vect_a, vect_b)
                dist_arr[i, j] = dist
                dist_arr[j, i] = dist
                j += 1

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
                
                # Get coordinates for vertex idx_a
                cp_a = np.array([graph.vertices[idx_a].loc_x, graph.vertices[idx_a].loc_y])
                

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
                    cp_b = np.array([graph.vertices[idx_b].loc_x, graph.vertices[idx_b].loc_y])
                    cp_diff = graph.distance(cp_a, cp_b)
                    if cp_diff == 0: 
                        continue

                    dist_error = (cp_diff - dist_target) / (2 * cp_diff)    
    
                    # Calculate coordinate update step sizes
                    if dist_target != 0: 
                        upd_scaling = (dist_target ** -2) * lr
                    else: 
                        upd_scaling = 1 
                    if upd_scaling > 1: 
                        upd_scaling = 1                    
                    dist_upd = (cp_a - cp_b) * dist_error * upd_scaling

                    # Update the vertex coordinatex
                    graph.vertices[idx_a].loc_x -= dist_upd[0]   
                    graph.vertices[idx_a].loc_y -= dist_upd[1]   
                    graph.vertices[idx_b].loc_x += dist_upd[0] 
                    graph.vertices[idx_b].loc_y += dist_upd[1] 

                    # Calculate stress 
                    stress += math.fabs(cp_diff - dist_target)
                
                # Progress monitoring
                if verbose: 
                    a_loop += 1
                    self.progress_print(iter_round, a_loop, stress)
                
            stress_list.append(stress)
        return stress_list
    
    
    def get_min_max_coords(self, graph):
        """
        Helper function to get graph vertex location min max coordinates.
        """
        # Get min and max coordinates of the graph vertices
        x_min = graph.vertices[0].loc_x
        x_max = graph.vertices[0].loc_x
        y_min = graph.vertices[0].loc_y
        y_max = graph.vertices[0].loc_y
        for vertex in graph.vertices:
            if x_min > vertex.loc_x:
                x_min = vertex.loc_x
            if x_max < vertex.loc_x:
                x_max = vertex.loc_x
            if y_min > vertex.loc_y:
                y_min = vertex.loc_y                
            if y_max < vertex.loc_y:
                y_max = vertex.loc_y 
                
        return x_min, x_max, y_min, y_max

    def plot_thumbnail(self, canvas, thumbnail, loc_x, loc_y, radius):
        """
        Plot vertex thumbnail on the canvas. 
        
        Args: 
            canvas : the plotting canvas
            thumbnail : the vertex image
            loc_x : vertex center point x location in pixels
            loc_y : vertex center point y location in pixels
            radius : vertex radius in pixels
            
        Returns: 
            the canvas where the image has been painted. 
        """
        
        # Check that we are not plotting outside of the canvas
        if (loc_x - radius < 0) or (loc_y - radius < 0):
            return canvas
        if (loc_x + radius >= canvas.shape[1]) or (loc_y + radius >= canvas.shape[0]):
            return canvas
        
        # Create circular mask for the thumbnail plot
        tn_dim = radius * 2
        mask = np.zeros((tn_dim, tn_dim, 3), dtype=np.uint8)
        mask = cv2.circle(mask, (radius, radius), radius=radius, thickness=-1, color=[1, 1, 1])
        
        # Scale and mask the thumbnail image
        tn = cv2.resize(thumbnail.copy(), dsize=(tn_dim, tn_dim))
        tn *= mask
        
        # Erase the thumbnail area from the canvas
        mask = ((mask == 0) * 1).astype(np.uint8)
        canvas[loc_y - radius : loc_y - radius + tn_dim, 
               loc_x - radius : loc_x - radius + tn_dim] *= mask
        
        # Copy thumbnail to the canvas
        canvas[loc_y - radius : loc_y - radius + tn_dim, 
               loc_x - radius : loc_x - radius + tn_dim] += tn 

        return canvas



    def plot_graph(self, graph, title='', 
                    vertex_radius=10, 
                    path_list=None, 
                    pad_ratio=0.1,
                    img_x=2000,
                    img_y=2000,
                    figsize = (13, 13),
                    background_color=[255, 255, 255],
                    plot_edges=False, 
                    plot_path=False,
                    plot_images=False,
                    save=False, filename=None):
        """
        Make 2D plot of the graph.
        
        Args: 
            graph : the graph to be plotted
            title : string to print on top middle of the plot. 
            path_list : path between graph vertices to be highlighted. 
            pad_ratio : padding (edges) around the graph as ratio of the graph size. 
            img_x : image width in pixels.
            img_y : image height in pixels.
            figsize : displayed image size. 
            background_color = RGB color of the plot background
            plot_edges : boolean, set True to plot edges, False to not plot.
            plot_path : boolean, set True to plot the path. 
            plot_images : boolean, set True to plot vertex images.
            save : boolean, set True to save the plot. 
            filename : file path where to save the plot if it is to be saved.
        """
        
        text_color = [0, 0, 0]
        line_color = [0, 0, 0]
        path_color = [255, 0, 0]


        # Create canvas in 8-bit format and set the background color
        canvas = np.zeros((img_y, img_x, 3), dtype=np.uint8) 
        canvas[:, :] = background_color
        
        # Get min and max coordinates
        x_min, x_max, y_min, y_max = self.get_min_max_coords(graph)

        # Add margins to canvas
        x_min -= pad_ratio * (x_max - x_min)
        x_max += pad_ratio * (x_max - x_min)
        y_min -= pad_ratio * (y_max - y_min)
        y_max += pad_ratio * (y_max - y_min)

        # Define coordinate mapping to canvas pixels
        x_step = (x_max - x_min) / img_x
        y_step = (y_max - y_min) / img_y 

        def scale_x(coord):
            return int((coord - x_min) / x_step)

        def scale_y(coord):
            return int((coord - y_min) / y_step)

        # Plot title
        cv2.putText(canvas, title, (100, 100), fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale=2, thickness=2, color=text_color)

        # Plot edges
        if plot_edges:
            for vertex in graph.vertices: 
                cp_x = scale_x(vertex.loc_x)
                cp_y = scale_y(vertex.loc_y)
                for edge_key in vertex.edges:
                    idx = graph.keys[edge_key]
                    edge_cp_x = scale_x(graph.vertices[idx].loc_x)
                    edge_cp_y = scale_y(graph.vertices[idx].loc_y)
                    cv2.line(canvas, (cp_x, cp_y), (edge_cp_x, edge_cp_y), 
                             thickness=1, lineType=cv2.LINE_AA, color=line_color)

        # Draw path
        if plot_path:
            for i in range(len(path_list) - 1): 
                idx_a = graph.keys[path_list[i]]
                idx_b = graph.keys[path_list[i + 1]]
                a_cp_x = scale_x(graph.vertices[idx_a].loc_x)
                a_cp_y = scale_y(graph.vertices[idx_a].loc_y)
                b_cp_x = scale_x(graph.vertices[idx_b].loc_x)
                b_cp_y = scale_y(graph.vertices[idx_b].loc_y)
                cv2.line(canvas, (a_cp_x, a_cp_y), (b_cp_x, b_cp_y), 
                         thickness=4, lineType=cv2.LINE_AA, color=[255, 0, 0])

        # Plot vertices 
        for vertex in graph.vertices: 
            cp_x = scale_x(vertex.loc_x)
            cp_y = scale_y(vertex.loc_y)

            if plot_images: 
                canvas = self.plot_thumbnail(canvas, vertex.image, cp_x, cp_y, vertex_radius)
            else:
                cv2.circle(canvas, (cp_x, cp_y), radius=vertex_radius, thickness=-1, color=vertex.color)
            
            # Draw  border around the thumbnail or colored vertex
            if plot_path and (vertex.key in path_list): 
                color = path_color
            else:
                color = line_color
            cv2.circle(canvas, (cp_x, cp_y), radius=vertex_radius, 
                        lineType=cv2.LINE_AA, thickness=2, color=color)




            #cv2.putText(canvas, str(vertex.key), (cp_x + 20, cp_y), fontFace=cv2.FONT_HERSHEY_DUPLEX, 
            #            fontScale=1, thickness=3, color=[0, 0, 0])

        plt.figure(figsize=figsize)
        plt.imshow(canvas, interpolation='sinc')
        plt.axis('off')
        plt.show()

        if save:
            cv2.imwrite(filename, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


        
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
        
        if (not key_a in graph.keys) or (not key_b in graph.keys):
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



