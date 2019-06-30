# Graph tools for Machine Learning 

This repository is a collection of graph tools for Machine Learning. 

#### FeatureGraph framework:
* Lightweight graph framework for constructing a graph and doing elementary operations with it. It is oriented but not limited to use with feature vectors, word vectors and similar vectorized data. The graph vertices can have any kind of user-defined properties, such as:
  * x, y location for plotting the graph
  * Graph edges (connections) and their weights (feature vector distances or user defined distance)
  * Color or image thumbnail for visualizing the graph
* Graph layout optimization using Stochastic Gradient Descent algorithm.
  * Sammon Mapping (https://en.wikipedia.org/wiki/Sammon_mapping) from high-dimensional space to 2-dimensional coordinates. It tries to preserve both local and global structures in the data. 
  * More 'traditional' edge distance stress based mapping algorithm. 
  * Both algorithms are adapted from paper https://arxiv.org/pdf/1710.04626.pdf 
* Delaunay triangulation graph edge definition. 
  * This can be done based on e.g. vertex locations or feature vectors. 
  * The edges can be also defined manually. 
* Edge weighted Dijkstra's algorithm for finding shortest path between graph vertices.
* Graph plotting 
  * Plot the graph using pre-defined colors, thumbnail image for vertices, etc. 
  * Highlight paths in the graph. 


#### Few usage notes: 
* Layout optimization: 
  * The learning rate (lr) value will have to be manually optimized for the feature vector data. A good starting point is to use lr value in the same ballpark with the feature vector standard deviation value. Lr value of 10 works fine with ResNet feature vectors, but sparse TFIDF vectors may need lr values around 0.001. 
  * With 2-dimensional feature vectors the mapping works directly and often results in perfect layout. With dense high-dimensional vectors it often does best job when the vector length is limited to max few tens using e.g. PCA. It could be used with e.g. 2048 long vectors from ResNet-50, but the projection often struggles to separate any clusters, since many of the feature sub-spaces are overlapping. 
  * The algorithm has O(n^2) computing complexity (n = number of vertices), so it will take some time to optimize graphs with large number of vertices. Feature vector length does not have significant impact on the computing time. 
  * There are some special cases that are difficult for this algorithm, for example, long narrow ribbon of vertices that may result in the layout having twists in the ribbon. If often helps to run the algorithm few iterations with large learning rate (lr) to shake those things towards better optimization. 
* Delaunay triangulation edge definition: 
  * The complexity skyrockets with large feature vector dimensions, so the Delaunay method works best when feature vectors are max about 8-dimensional. With high dimensional vectors another drawback is that the vertices are connected to almost every other vertex, so the graph processing can become heavy. 


## Animal face graph notebook
This notebook shows simple example how to construct a graph based on image feature vectors (from ResNet-50) and how to do some fun and probably useless things with it. Total approx. 70 images of dogs, horses, ducks and cats were used to create the example. 



(c) Mikko Kursula 2018 - 2019. MIT License. 