import numpy as np
from kmeans import pairwise_dist
class DBSCAN(object):
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset
    def fit(self):
        """Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        Hint: Using sets for visitedIndices may be helpful here.
        Iterate through the dataset sequentially and keep track of your points' cluster assignments.
        If a point is unvisited or is a noise point (has fewer than the minimum number of neighbor points), then its cluster assignment should be -1.
        Set the first cluster as C = 0
        """
        C = 0
        visitedIndices = set()
        cluster_idx = np.ones((self.dataset.shape[0])) * -1
        #print('visited indices set: ', visitedIndices)
        #print('initial cluster ids: ', cluster_idx)
        #print(self.dataset.shape)
        for point_i in range(0, self.dataset.shape[0]):
            if point_i not in visitedIndices:
                visitedIndices.add(point_i)
                neighborIndices = self.regionQuery(point_i)
                if neighborIndices.size < self.minPts:
                    cluster_idx[point_i] = -1
                    #print('I am an outlier')
                else:
                    #print(C)
                    self.expandCluster(point_i, neighborIndices, C, cluster_idx, visitedIndices)
                    C += 1
        #print(cluster_idx)
        #print(len(visitedIndices))
        #print(visitedIndices)
        return cluster_idx

    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly
           HINT: regionQuery could be used in your implementation
        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
            C: current cluster as an int
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        Hints: 
            np.concatenate(), np.unique(), np.sort(), and np.take() may be helpful here
            A while loop may be better than a for loop
        """
        #print(index)
        #print(neighborIndices)
        #print(C)
        #print(cluster_idx)
        #print(visitedIndices)

        #Add P to cluster C
        cluster_idx[index] = C
        #print(cluster_idx)

        i = 0
        while i < len(neighborIndices):
            curr_point_i = neighborIndices[i]
            #print('current point index: ', curr_point_i)
            if curr_point_i not in visitedIndices:
                #Mark P as visited
                visitedIndices.add(curr_point_i)
                #Extract the neighboirs' indexes of the current point
                curr_point_neighbors = self.regionQuery(curr_point_i)
                #print('current point neighbors:', curr_point_neighbors)
                #print(curr_point_neighbors.size)
                if curr_point_neighbors.size >= self.minPts:
                    neighborIndices = np.concatenate((neighborIndices, curr_point_neighbors))
                    #print(neighborIndices)
            if cluster_idx[curr_point_i] == -1:
                #print('no hola')
                cluster_idx[curr_point_i] = C
            i += 1
        #print('expand clusters id assignment: ', cluster_idx)
        #print(visitedIndices)
        return

    def regionQuery(self, pointIndex):
        """Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
        Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
        """
        #print(pointIndex)
        #print(self.eps)
        curr_point = self.dataset[pointIndex].reshape(1, self.dataset.shape[1])
        #print(curr_point.shape)
        #print(self.dataset.shape)
        eu_distances = pairwise_dist(curr_point,self.dataset)
        #print(eu_distances)
        indices = np.where(eu_distances <= self.eps)[1]
        #print('indices: ', indices)
        return indices
        