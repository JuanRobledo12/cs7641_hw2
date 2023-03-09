
'''
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
'''

import numpy as np


class KMeans(object):

    def __init__(self, points, k, init='random', max_iters=10000, rel_tol=1e-5):  # No need to implement
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria w.r.t relative change of loss
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == 'random':
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):# [2 pts]
        """
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """

        centers = self.points[np.random.choice(self.points.shape[0], self.K, replace=False), :]
        return centers

    def kmpp_init(self):# [3 pts]
        """
            Use the intuition that points further away from each other will probably be better initial centers
        Return:
            self.centers : K x D numpy array, the centers.
        """
        #print(self.K)
        #print(self.points.shape[0])
        sample_size = int(self.points.shape[0] * 0.01)
        samples = self.points[np.random.choice(self.points.shape[0], sample_size, replace=False), :]
        #print(samples) 
        #print('---------------------')
        centers = samples[np.random.choice(samples.shape[0], 1), :]
        #print(centers)
        #print(centers.shape)
        #print('---------------------')
        

        ## DEAL WITH FINDING THE MINIMUM DISTANCE OF THE MAX VALUES IN EACH CENTER. Check Instruciton 3

        for k in range(0, self.K - 1):
            #print('K_i: ', k)
            sq_dist_arr = pairwise_dist(centers, samples)**2
            #print(eu_dist_arr)
            #print(eu_dist_arr.shape)
            #print('---------------------')
            #---------------Frist Approach -------------------------
            # max_value_arr = np.amax(eu_dist_arr, axis=1)
            # print('max values: ', max_value_arr)
            # print('---------------------')
            # #print(samples[max_value_i, :].shape)
            # min_cluster_i = np.argmin(max_value_arr)
            # print('index of min dist value: ', min_cluster_i)
            # print('---------------------')
            # max_value_i = np.argmax(eu_dist_arr[min_cluster_i,:])
            # print('index of the max distance from the min cluster: ', max_value_i)
            # print('---------------------')
            
            #---------------Second Approach -------------------------
            min_dist_arr = np.min(sq_dist_arr, axis=0)
            #print('min_dist_arr: ', min_dist_arr)
            #print('---------------------')
            max_value = np.amax(min_dist_arr)
            #print('max_value: ', max_value)
            #print('---------------------')
            max_value_i = np.where(sq_dist_arr == max_value)[1][0]
            #print('max_value_i: ', max_value_i)
            #print('---------------------')



            #-------------------------------------------------
            new_center = samples[max_value_i, :]
            #print('new center: ', new_center)
            #print('---------------------')
            centers = np.append(centers, new_center.reshape(1, new_center.shape[0]), axis=0)
            #print(centers)
            #print('---------------------')
            
        return centers

    def update_assignment(self):  # [5 pts]
        """
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: You could call pairwise_dist() function
        """        
        #print(self.centers)
        #print('----------------------')
        #print(self.points)
        #print('----------------------')

        curr_centers = self.centers
        curr_datapoints = self.points

        eu_dist_arr = pairwise_dist(curr_centers, curr_datapoints)
        #print(eu_dist_arr)
        #print('----------------------')
        self.assignments = np.argmin(eu_dist_arr, axis=0)
        #print(assignments)
        #print('----------------------')


        #eu_dist_arr = pairwise_dist(self.centers, self.points)
        
        #print(eu_dist_arr)
        return self.assignments

    def update_centers(self):  # [5 pts]
        """
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        """
        
        datapoints = self.points
        cluster_ids = self.assignments
        k = self.K
        old_centers = self.centers
        #print('datapoints: ', datapoints)
        #print('-----------------------')
        #print('cluster ids: ', cluster_ids)
        #print('--------------------')

        new_centers = np.zeros_like(old_centers, dtype=float)
        #print('New Centers: ', new_centers)
        #print('--------------------')

        for i in range(0,k):
            #print('K_i: ', i)
            cluster_points = datapoints[cluster_ids == i]
            #print('Cluster points: ', cluster_points)
            #print('--------------------')
            mean = np.mean(cluster_points, axis=0)
            new_centers[i] = mean

        #print('Final New Centers: ', new_centers)
        
        return new_centers

    def get_loss(self):  # [5 pts]
        """
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """
        curr_centers = self.centers
        datapoints = self.points
        cluster_ids = self.assignments
        k = self.K

        sq_dist_arr = np.zeros((1, k), dtype=float)
        #print(sq_dist_arr)

        for i in range(0, k):
            #print('k: ', i)
            cluster_points = datapoints[cluster_ids == i]
            #print('cluster points: ', cluster_points)
            #print('--------------------')
            #print('cluster center: ',curr_centers[i, :].reshape(1, curr_centers.shape[1]))
            #print('--------------------')
            eu_dist = pairwise_dist(curr_centers[i, :].reshape(1, curr_centers.shape[1]), cluster_points)
            #print('eu_dist', eu_dist)
            #print('--------------------')
            #print(np.sum(eu_dist**2))
            sq_dist_arr[0, i] = np.sum(eu_dist**2)
            #print(sq_dist_arr)
        
        loss = np.sum(sq_dist_arr)
        #print('Loss', loss)
        return loss
    def train(self):    # [10 pts]
        """
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster, 
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned, 
                     pick a random point in the dataset to be the new center and 
                     update your cluster assignment accordingly.
                4. Calculate the loss and check if the model has converged to break the loop early.
                   - The convergence criteria is measured by whether the percentage difference 
                     in loss compared to the previous iteration is less than the given 
                     relative tolerance threshold (self.rel_tol). 
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!
                
        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.
        """
        #print('----------------------------------')
        #print('max_iters:', self.max_iters)
        prev_loss = 100000
        
        for i in range(0, self.max_iters):
            #print(i)
            self.assignments = self.update_assignment()
            self.centers = self.update_centers()
            center_list = np.arange(self.K)
            used_clusters = np.unique(self.assignments)
            #print(used_clusters)
            #print(np.array_equal(center_list, used_clusters))
            #Check to make sure there is no cluster center without any points assigned to it
            if not np.array_equal(center_list, used_clusters):
                empty_cluster_arr = np.setdiff1d(center_list, used_clusters)
                #Find the index in center list of the missing values.
                #With this index replace the row at self.centers with a random row in points.
                #Probably we will use a for loop to replace multiple rows.
                for cluster_i in empty_cluster_arr:
                    new_center = self.points[np.random.choice(self.points.shape[0], 1, replace=False), :]
                    self.centers[cluster_i] = new_center
                    #print('Im inside the for loop replaceing this cluster: ', cluster_i)
            #print('prev_loss: ', prev_loss)
            self.loss = self.get_loss()
            loss_diff = np.abs(prev_loss - self.loss) / prev_loss
            prev_loss = self.loss
            #print('loss: ', self.loss)
            #print('loss_diff', loss_diff)
            #print('tolerance', self.rel_tol)
            if loss_diff < self.rel_tol:
                #print('I am breaking free!!')
                break
        return self.centers, self.assignments, self.loss

def pairwise_dist(x, y):  # [5 pts]
        np.random.seed(1)
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
        """
        #Sum the square of X and Y across the dimensions
        x_2 = np.sum(x**2, axis=1)
        y_2 = np.sum(y**2, axis=1)

        #Multiply X and Y
        X_Y = x@np.transpose(y)

        #Calculate the pairwise euclidean distance
        eu_dist = np.sqrt(np.abs(x_2.reshape(x_2.shape[0], 1) + y_2 - (2*X_Y)))

        return eu_dist
