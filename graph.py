import numpy as np

from graphutils import GraphUtilsMixin
from graph_layout import GraphLayoutMixin


class GraphVertex:
    def __init__(self, key, loc_x, loc_y, 
                 color, labels):
        self.key = key
        self.labels = labels
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.color = color
        self.image = None

        """
        edges_in and edges_out will store edges in {key : edge} pairs, 
        where key is the source or target vertex key (respectively)
        and edge object contains the edge params. 
        """
        self.edges_out = {} # edges originating from this vertex. 
        self.edges_in = {} # edges pointing to this vertex. 

        # All app specific params go to the params dict. 
        self.params = {}


class GraphEdge:
    """
    Graph edge data structure. 
    """
    def __init__(self, distance, bidir, params, labels):
        self.distance = distance
        self.bidir = bidir
        self.labels = []
        self.params = params 

class Graph(GraphUtilsMixin, GraphLayoutMixin):
    """
    Graph class containing all functions closely related to graph construction. 
    """
    
    def __init__(self):
        """
        Initialize the graph. Basically it is just dict of vertices. 
        All edges are defined inside the vertices. 
        """
        self.vertices = {}


    def add_vertex(self, key, loc_x=0, loc_y=0, color=[0, 0, 0], labels=[None]):
        """
        Adds vertex to the graph.
        Args: 
            key : identifier for the vertex. Can be e.g. integer or string. 
            loc_x, loc_y : location for plotting purposes.
            color : vertex plot color.
            labels : vertex type labels list. 
        """
        if key in self.vertices:
            raise ValueError('Key is already in use')
        
        # Create vertex
        self.vertices[key] = GraphVertex(key=key, color=color, 
                                         loc_x=loc_x, loc_y=loc_y, 
                                         labels=labels)        


    def add_vertex_image(self, key, image):
        """
        Add thumbnail image to the vertex (for plotting purposes only)
        """
        if not key in self.vertices:
            raise ValueError('Key not found')

        self.vertices[key].image = image


    def add_vertex_params(self, key, **kwargs):
        """
        Add application specific params to the vertex. 
        """
        for param_key, param_value in kwargs.items():
            self.vertices[key].params[param_key] = param_value


    def delete_vertex(self):
        """
        FIXME 
        """
        pass


    def add_edge_params(self, key_a, key_b, **kwargs):
        # FIXME
        pass
        

    def add_edge(self, key_a, key_b, distance=1, bidirectional=True, params={}, labels=[None]):
        """
        Add edge to the graph from vertex key_a to key_b. If bidirectional is set, 
        adds also the opposite direction edge, since bidirectional edge is 
        two directional edges in opposite directions.
        """
        # Check that both keys exist
        if (not key_a in self.vertices) or (not key_b in self.vertices):
            raise ValueError('Key not found')
                
        # Add edge to the vertex lists
        edge = GraphEdge(distance, bidirectional, params, labels)
        self.vertices[key_a].edges_out[key_b] = edge 
        self.vertices[key_b].edges_in[key_a] = edge 
        if bidirectional:
            self.vertices[key_b].edges_out[key_a] = edge 
            self.vertices[key_a].edges_in[key_b] = edge 

    
    def delete_edge(self, key, all=True):
        if not key in self.vertices:
            raise ValueError('Key not found')

        if all:
            self.vertices[key].edges_in = {}
            self.vertices[key].edges_out = {}

            # FIXME: delete also the other end edge instances. 

        else: 
            # FIXME: delete single edge
            pass

          
    def save(self, filename):
        np.save(filename, self.vertices)
    
    
    def load(self, filename):
        self.vertices = np.load(filename).item()


