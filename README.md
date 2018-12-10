# Graph tools for Machine Learning 

This repository is a collection of graph tools for Machine Learning. 

#### Currently available features:
* Lightweight graph framework for constructing a graph and doing elementary operations with it. It is oriented for use with feature vectors, word vectors and similar vectorized data. The graph vertices can have properties like:
  * Feature vector
  * x, y location for plotting the graph
  * Graph edges (connections) and their weights (feature vector distances)
  * Color or image thumbnail for plotting the graph
  * Text description that can be used to store e.g. file path. 
* Graph layout optimization using Stochastic Gradient Descent algorithm.
  * The algorithm is adapted from https://arxiv.org/pdf/1710.04626.pdf 
  * It essentially does Sammon Mapping (https://en.wikipedia.org/wiki/Sammon_mapping) operation from high-dimensional space to 2-dimensional coordinates. It tries to preserve both local and global structures in the data. 
* Delaunay triangulation graph edge definition. 
  * This can be done based on the vertex locations or feature vectors. 
  * The edges can be also defined manually. 
* Edge weighted Dijkstra's algorithm for finding shortest path between graph vertices.
* Graph plotting 
  * Can plot the vertices using pre-defined color or thumbnail image. 
  * Can highlight paths in the graph. 


#### Few usage notes: 
* Treat the vertex feature vectors as immutable after any edges are defined for that vertex. The edge distances (feature vector distances) are calculated based on the feature vectors when adding the edges and those won't be updated if the feature vector is modified. 
* Layout optimization: 
  * With 2-dimensional feature vectors the mapping works directly and often results in perfect layout. With dense high-dimensional vectors it often does best job when the vector length is limited to max few tens using e.g. PCA. It could be used with e.g. 2048 long vectors from ResNet-50, but the projection often struggles to separate any clusters, since many of the feature sub-spaces are overlapping. 
  * The algorithm has O(n^2) computing complexity (n = number of vertices), so it will take some time to optimize graphs with large number of vertices. Feature vector length does not have significant impact on the computing time. 
  * There are some special cases that are difficult for this algorithm, for example, long narrow ribbon of vertices, which may result in the layout having twists in the ribbon. If often helps to run the algorithm few iterations with large learning rate (lr) to shake those things towards better optimization. 
* Delaunay triangulation edge definition: 
  * The complexity skyrockets with large feature vector dimensions, so the Delaunay method works best when feature vectors are max about 8-dimensional. With high dimensional vectors another drawback is that the vertices are connected to almost every other vertex, so the edges sort of lose their meaning. 
  * The most generic solution is to first run the layout optimization and then perform the triangulation based on the vertex x,y locations. 


## Animal face graph notebook
This notebook shows simple example how to construct a graph based on image feature vectors and how to do some fun and probably useless things with it. Total approx. 70 images of dogs, horses, ducks and cats were used to create the example. 



(c) Mikko Kursula 2018. MIT License. 