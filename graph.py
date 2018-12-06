import numpy as np


class GraphVertex(object):
    def __init__(self, key, feature_vector, loc_x=0, loc_y=0, 
                 color=[0, 0, 0], description=None, image=None):
        """
        Graph vertex data structure. 
        
        Args: 
            key : identifier for the vertex. Can be e.g. integer or string. 
            feature_vector : 1D Numpy array containing the feature vector. 
            loc_x, loc_y : vertex coordinate location in 2D plot. 
            color : vertex plot color.
            description : description data, e.g. image file path. 
            image : thumbnail image Numpy array to be used in graph plotting. 
        
        """
        self.key = key
        self.feature_vector = feature_vector.astype(np.float32)
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.color = color
        self.description = description
        self.image = image
        self.edges = []
        self.distances = []

class Graph(object):
    """
    Graph class containing all functions closely related to graph construction. 
    """
    
    def __init__(self):
        """
        Initialize the graph.
        """
        self.vertices = [] # list to store vertex instances
        self.keys = {} # dict to map key values to vertices list indexes. 


    def add_vertex(self, key, feature_vector, color=[0, 0, 0], 
                    image=None, description=None):
        """
        Adds vertex to the graph.
        
        Args: 
            key : identifier for the vertex. Can be e.g. integer or string. 
            feature_vector : 1D Numpy array containing the feature vector. 
            color : vertex plot color.
            image : thumbnail image Numpy array to be used in graph plotting. 
            description : description data, e.g. image file path. 
            
        Returns:
            nothing
        """
        if key in self.keys:
            raise ValueError('Key is already in use')
        # Add vertex idx to key dictionary
        idx = len(self.vertices)
        self.keys[key] = idx
        
        # Create vertex
        self.vertices.append(GraphVertex(key, feature_vector, 
                                         color=color, 
                                         image=image,
                                         loc_x=np.random.rand(),
                                         loc_y=np.random.rand(),
                                         description=description))
        
    
    def delete_vertex(self):
        """
        TODO
        """
        pass
    
    
    def delete_edge(self):
        """
        TODO
        """
        pass
    
        
    def add_edge(self, key_a, key_b):
        """
        Add edge to the graph between two vertices. 
        
        Args:
            key_a : key of the first vertex.
            key_b : key of the second vertex.
        
        Returns:
            Nothing
        """
        # Check that both keys exist
        if (not key_a in self.keys) or (not key_b in self.keys):
            raise ValueError('Key not found')

        # Get vertex list indices
        idx_a = self.keys[key_a]
        idx_b = self.keys[key_b]
        
        # Check if edge exists
        if key_b in self.vertices[idx_a].edges:
            return 
        
        # Calculate distance
        vect_a = self.vertices[idx_a].feature_vector
        vect_b = self.vertices[idx_b].feature_vector
        dist = self.distance(vect_a, vect_b)
        
        # Add edge to the vertex lists
        self.vertices[idx_a].edges.append(key_b)
        self.vertices[idx_a].distances.append(dist)
        self.vertices[idx_b].edges.append(key_a)
        self.vertices[idx_b].distances.append(dist)
        

    def get_description(self, key):
        if not key in self.keys:
            raise ValueError('Key not found')

        return self.vertices[self.keys[key]].description


    def get_path_dist(self, key_list):
        """  
        Get the total feature vector distance on a given path in the graph. 
        
        Args: 
            key_list : indexable list of vertex keys that define the path 
            in the graph. The graph must contain edges that correspond to the 
            path. 
            
        Returns: 
            Total distance value. 
        """        
        dist = 0 
        for i in range(len(key_list) - 1):
            key_a = key_list[i]
            key_b = key_list[i + 1]     
            idx_a = self.keys[key_a]
            for edge_idx in range(len(self.vertices[idx_a].edges)):
                if self.vertices[idx_a].edges[edge_idx] == key_b:
                    dist += self.vertices[idx_a].distances[edge_idx]
                    break
        return dist
    
    
    def get_edges(self, key):
        """
        Get keys of all the vertices that are connected to the
        vertex with given key. 
        
        Args: 
            key : Key of the vertex of which the edges are reported. 
            
        Returns:
            List of keys of the directly connected vertices.
        """
        if not key in self.keys:
            raise ValueError('Key not found')
            
        idx = self.keys[key]
        return self.vertices[idx].edges


    def distance(self, vect_a, vect_b):
        """
        Calculates Euclidean distanve between two vectors. 

        Args:
            vect_a : feature vector 
            vect_b : feature vector 
            
        Returns: 
            Euclidean distance
        """
        dist = np.linalg.norm(vect_a - vect_b)
        return dist
        
    
    def save_graph(self):
        """
        TODO
        """
        pass
    
    
    def load_graph(self):
        """
        TODO
        """
        pass