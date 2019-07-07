import numpy as np
import matplotlib.pyplot as plt
import cv2

from .graph import Graph 

class GraphPlot: 
    
    def __init__(self, 
                 img_x : int = 2000, 
                 img_y : int = 2000, 
                 plot_size : tuple = (13, 13),
                 pad_ratio : float = 0.1,
                 background_color : list = [255, 255, 255],
                 highlight_color : list = [255, 10, 10],
                 default_edge_color : list = [0, 0, 0],
                 default_vertex_color : list = [0, 0, 0], 
                 default_vertex_radius : float = 1.0, 
                 default_line_width : float = 0.1,
                 highlight_line_width : float = 0.2 
                 ):

        self.img_x = img_x
        self.img_y = img_y
        self.plot_size = plot_size
        self.pad_ratio = pad_ratio
        self.background_color = background_color
        self.highlight_color = highlight_color
        self.highlight_line_width = highlight_line_width
        self.default_edge_color = default_edge_color
        self.default_vertex_color = default_vertex_color
        self.default_vertex_radius = default_vertex_radius # this is in pixels
        self.default_line_width = default_line_width
        
        # initialize the plot canvas
        self.canvas = np.zeros((img_y, img_x, 3), dtype=np.uint8) 
        self.canvas[:, :] = self.background_color

        
    def set_graph(self, graph : Graph) -> None:
        """
        Load graph to the plot object. 
        """
        self.graph = graph
        self.__get_graph_min_max_coords()
        self.__add_padding()
        self.__calculate_scaling_params()
        self.__calculate_canvas_center_point()

        self.highlight_vertices = []
        self.highlight_edges = []
        
        
    def plot_title(self, 
                   title : str, 
                   loc : tuple = (100, 100), 
                   scale : int = 2, 
                   thickness : int = 2, 
                   color : list = [0, 0, 0]) -> None:
        """
        Plot title string on the canvas.
        """
        cv2.putText(self.canvas, title, loc, 
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale=scale, thickness=thickness, color=color)
        
        
    def plot_edges(self) -> None:
        """
        Plot all edges in the graph. 
        """
        vertices = self.graph.get_vertices()

        for key_a in vertices: 
            edge_keys = self.graph.get_edges(key_a)['out']
            for key_b in edge_keys:                
                self.__plot_edge(key_a, key_b)
                
                
    def __plot_edge(self, 
                    key_a : str, 
                    key_b : str):

        loc_a = self.graph.get_vertex_param(key_a, 'location')
        loc_a = self.__scale_loc(loc_a)
        
        loc_b = self.graph.get_vertex_param(key_b, 'location')
        loc_b = self.__scale_loc(loc_b)


        width = self.graph.get_edge_param(key_a, key_b, 'width')
        if width is None: 
            width = self.default_line_width

        # Convert width dimension to pixel 

        color = self.graph.get_edge_param(key_a, key_b, 'color')
        if color is None: 
            color = self.default_edge_color

        if ([key_a, key_b] in self.highlight_edges) or \
            ([key_b, key_a] in self.highlight_edges):
            cv2.line(self.canvas, 
                     loc_a, loc_b, 
                     thickness=self.__scale_dim(self.highlight_line_width), 
                     lineType=cv2.LINE_AA, 
                     color=self.highlight_color)
        else:
            cv2.line(self.canvas, 
                     loc_a, loc_b, 
                     thickness=self.__scale_dim(width), 
                     lineType=cv2.LINE_AA, 
                     color=color)
                        
                    
    def plot_vertices(self, plot_thumbnails : bool = False) -> None:
        """
        Plot all vertices of the graph
        """
        vertices = self.graph.get_vertices()
        for key in vertices: 
            self.__plot_vertex(key, plot_thumbnails)


    def __plot_vertex(self, key, plot_thumbnails):

        # Get location
        loc = self.graph.get_vertex_param(key, 'location')
        loc = self.__scale_loc(loc)

        # Get radius
        radius = self.graph.get_vertex_param(key, 'radius')
        if radius is not None: 
            radius = self.__scale_dim(radius)
        else: # no radius was found
            radius = self.__scale_dim(self.default_vertex_radius)

        # Get color
        color = self.graph.get_vertex_param(key, 'color')
        if color is None: 
            color = self.default_vertex_color

        if plot_thumbnails: 
            image = self.graph.get_vertex_param(key, 'image')
            self.__plot_thumbnail(image, loc[0], loc[1], radius)
        else: # draw solid circle
            cv2.circle(self.canvas, loc, radius=radius, 
                       thickness=-1, color=color)
            
        # Draw  border around the thumbnail or colored vertex
        if key in self.highlight_vertices: 
            cv2.circle(self.canvas, 
                       loc, 
                       radius=radius, 
                       lineType=cv2.LINE_AA, 
                       thickness=self.__scale_dim(self.highlight_line_width), 
                       color=self.highlight_color)
        else:
            cv2.circle(self.canvas, 
                       loc, 
                       radius=radius, 
                       lineType=cv2.LINE_AA, 
                       thickness=self.__scale_dim(self.default_line_width), 
                       color=self.default_edge_color)

            
    def highlight_path(self, path : list):
        """
        Highlight the path in the graph plot. 
        This has to be done before plotting the edges or vertices. 
        """
        for i in range(len(path) - 1): 
            key_a = path[i]
            key_b = path[i + 1]
            self.highlight_edges.append([key_a, key_b])
            self.highlight_vertices.append(key_a)
            self.highlight_vertices.append(key_b)
            

    def __scale_dim(self, dim):        
        int_dim = int(round(dim / self.pixel_size))
        if int_dim <= 0: 
            int_dim = 1
        return int_dim


    def __scale_loc(self, loc): 
        x = self.__scale_x(loc[0])
        y = self.__scale_y(loc[1])            
        return (x, y)


    def __scale_x(self, x_coord):
        x_index = round((x_coord - self.cp_x_coord) / self.pixel_size)
        x_index += self.cp_x_index
        x_index = int(x_index)
        return x_index
    

    def __scale_y(self, y_coord):
        y_index = round((y_coord - self.cp_y_coord) / self.pixel_size)
        y_index *= -1  # lower values at bottom
        y_index += self.cp_y_index
        y_index = int(y_index)
        return y_index


    def __calculate_canvas_center_point(self): 
        """
        Calculate center point that is the reference point for all drawing 
        operations. 
        """
        self.cp_x_coord = (self.x_min + self.x_max) / 2
        self.cp_y_coord = (self.y_min + self.y_max) / 2

        self.cp_x_index = round(self.img_x / 2)
        self.cp_y_index = round(self.img_y / 2)


    def __calculate_scaling_params(self):
        """
        Define the scaling coefficient for mapping coordinates to 
        image pixel indexes. 
        This assumes square pixels and defines equal pixel stepping 
        for x and y directions. 
        """
        x_pix_size = (self.x_max - self.x_min) / self.img_x
        y_pix_size = (self.y_max - self.y_min) / self.img_y

        if y_pix_size > x_pix_size: 
            self.pixel_size = y_pix_size
        else: 
            self.pixel_size = x_pix_size


    def __add_padding(self):
        """
        Add padding to the coordinate ranges. This effectively produces
        empty margins to the final image. 
        """
        self.x_min -= self.pad_ratio * (self.x_max - self.x_min)
        self.x_max += self.pad_ratio * (self.x_max - self.x_min)
        self.y_min -= self.pad_ratio * (self.y_max - self.y_min)
        self.y_max += self.pad_ratio * (self.y_max - self.y_min)

        
    def __get_graph_min_max_coords(self):
        """
        Helper function to get graph vertex location min max coordinates.
        """
        some_key = list(self.graph.vertices.keys())[0]
        self.x_min = self.graph.vertices[some_key].params['location'][0]
        self.x_max = self.graph.vertices[some_key].params['location'][0]
        self.y_min = self.graph.vertices[some_key].params['location'][1]
        self.y_max = self.graph.vertices[some_key].params['location'][1]
        
        for vertex in self.graph.vertices.values():
            if self.x_min > vertex.params['location'][0]:
                self.x_min = vertex.params['location'][0]
            if self.x_max < vertex.params['location'][0]:
                self.x_max = vertex.params['location'][0]
            if self.y_min > vertex.params['location'][1]:
                self.y_min = vertex.params['location'][1]   
            if self.y_max < vertex.params['location'][1]:
                self.y_max = vertex.params['location'][1] 
        

    def __plot_thumbnail(self, thumbnail, loc_x, loc_y, radius):
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



