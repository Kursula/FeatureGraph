import numpy as np

from .graphutils import GraphUtilsMixin
from .graph_layout import GraphLayoutMixin


class GraphVertex:
    def __init__(self, key):
        self.key = key

        # Edges_in and edges_out will store edges in {key : edge} pairs, 
        # where key is the source or target vertex key
        # and edge object contains the edge params. 
        self.edges_out = {} # edges originating from this vertex. 
        self.edges_in = {} # edges pointing to this vertex. 

        # All app specific params go to the params dict. 
        self.params = {}


class GraphEdge:
    """
    Graph edge data structure. 
    """
    def __init__(self, distance, bidirectional):
        self.params = {'distance' : distance, 
                       'bidirectional' : bidirectional}


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


    def add_vertex(self, key):
        """
        Adds vertex to the graph.
        Args: 
            key : identifier for the vertex. Can be e.g. integer or string. 
        """
        if key in self.vertices:
            raise ValueError('Key is already in use')
        
        # Create vertex
        self.vertices[key] = GraphVertex(key=key)        


    def set_vertex_param(self, key, **kwargs):
        """
        Add application specific params to the vertex. 
        """
        for param_key, param_value in kwargs.items():
            self.vertices[key].params[param_key] = param_value


    def get_vertex_param(self, key, param_key):
        """
        Get parameters stored in the vertex. 
        """
        value = self.vertices[key].params[param_key]
        return value


    def delete_vertex(self, key):
        """
        Delete vertex and all edges connected to that vertex.
        """
        # Delete edges
        self.delete_edge(key, bidirectional=True)

        # Delete the vertex 
        self.vertices.pop(key)


    def add_edge(self, key_a, key_b, distance=1, bidirectional=True):
        """
        Add edge to the graph from vertex key_a to key_b. If bidirectional is set, 
        adds also the opposite direction edge, since bidirectional edge is made of
        two directional edges in opposite directions.
        """
        # Check that both keys exist
        if (not key_a in self.vertices) or (not key_b in self.vertices):
            raise ValueError('Key not found')
                
        # Add edge to the vertex lists
        edge = GraphEdge(distance, bidirectional)
        self.vertices[key_a].edges_out[key_b] = edge 
        self.vertices[key_b].edges_in[key_a] = edge 
        if bidirectional:
            self.vertices[key_b].edges_out[key_a] = edge 
            self.vertices[key_a].edges_in[key_b] = edge 

    
    def delete_edge(self, key_a, key_b=None, bidirectional=True):
        """
        Delete edge from vertex key_a to key_b. 
        If bidirectional is True, deletes also the opposite edge if it exists.
        
        If key_b is not set, deletes all edges to/from vertex key_a.
        """

        if key_a not in self.vertices.keys(): 
            return 

        if key_b is not None and key_b not in self.vertices.keys(): 
            return 

        if key_b is None: 
            # Delete all edges to/from key_a vertex
            for key_b in self.vertices[key_a].edges_in:
                self.vertices[key_b].edges_out.pop(key_a)
            for key_b in self.vertices[key_a].edges_out:
                self.vertices[key_b].edges_in.pop(key_a)
            self.vertices[key_a].edges_in = {}
            self.vertices[key_a].edges_out = {}
            return 

        # Delete the key_a -> key_b direction edge
        if key_b in self.vertices[key_a].edges_out.keys(): 
            self.vertices[key_a].edges_out.pop(key_b)
        if key_a in self.vertices[key_b].edges_in.keys(): 
            self.vertices[key_b].edges_in.pop(key_a)

        if bidirectional:
            # Delete the key_b -> key_a direction edge
            if key_a in self.vertices[key_b].edges_out.keys(): 
                self.vertices[key_b].edges_out.pop(key_a)
            if key_b in self.vertices[key_a].edges_in.keys(): 
                self.vertices[key_a].edges_in.pop(key_b)


    def set_edge_param(self, key_a, key_b, **kwargs):
        """
        Add parameters to the edge params dictionary. 
        The same edge instance is referenced from all edges_in and edges_out
        dicts, so it is enough to update the params to just one of them. 
        """
        for param_key, param_value in kwargs.items():
            self.vertices[key_a].edges_out[key_b].params[param_key] = param_value


    def get_edge_param(self, key_a, key_b, param_key):
        value = self.vertices[key_a].edges_out[key_b].params[param_key]
        return value

          
    def save(self, filename):
        np.save(filename, self.vertices)
    
    
    def load(self, filename):
        self.vertices = np.load(filename).item()


