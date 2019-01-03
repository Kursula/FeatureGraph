import numpy as np
import matplotlib.pyplot as plt
import cv2


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
        # FIXME: dealing with bidirectional edges to avoid duplicate lines. 

        for vertex_a in self.graph.vertices.values(): 
            for edge_key in vertex_a.edges_out.keys():
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
        
        # FIXME: use vertex radius param. 

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



