import numpy as np
import math

def dist(x, y):
    return math.sqrt(sum([(x[k]-y[k])**2 for k in range(len(x))]))

class KMeansClustering:
    """
    K-Means Clustering Model

    Args:
        n_clusters: Number of clusters(int)
    """

    def __init__(self, n_clusters, n_init=10, max_iter=1000, delta=0.001):

        self.n_cluster = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.delta = delta

    def init_centroids(self, data):
        idx = np.random.choice(
            data.shape[0], size=self.n_cluster, replace=False)
        self.centroids = np.copy(data[idx, :])

    def fit(self, data):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix(M data points with D attributes each)(numpy float)
        Returns:
            The object itself
        """
        if data.shape[0] < self.n_cluster:
            raise ValueError(
                'Number of clusters is grater than number of datapoints')

        best_centroids = None
        m_score = float('inf')

        for _ in range(self.n_init):
            self.init_centroids(data)

            for _ in range(self.max_iter):
                cluster_assign = self.e_step(data)
                old_centroid = np.copy(self.centroids)
                self.m_step(data, cluster_assign)

                if np.abs(old_centroid - self.centroids).total_sum() < self.delta:
                    break

            cur_score = self.evaluate(data)

            if cur_score < m_score:
                m_score = cur_score
                best_centroids = np.copy(self.centroids)

        self.centroids = best_centroids

        return self

    def e_step(self, data):
        """
        Expectation Step.
        Finding the cluster assignments of all the points in the data passed
        based on the current centroids
        Args:
            data: M x D Matrix (M training samples with D attributes each)(numpy float)
        Returns:
            Cluster assignment of all the samples in the training data
            (M) Vector (M number of samples in the train dataset)(numpy int)
        """
        #TODO

        cluster = []
        for i in data:
            minV = math.sqrt(sum([(i[k]-self.centroids[0][k])**2 for k in range(len(i))]))
            min_index = 0
            for j in range(0,len(self.centroids)):
                d = math.sqrt(sum([(i[k]-self.centroids[j][k])**2 for k in range(len(i))]))
                if(d < minV):
                    minV = d
                    min_index = j
            cluster.append(min_index)
        return cluster

    def m_step(self, data, cluster_assgn):
        """
        Maximization Step.
        Compute the centroids
        Args:
            data: M x D Matrix(M training samples with D attributes each)(numpy float)
            cluster_assign: Cluster Assignment
        Change self.centroids
        """
        #TODO
        self.centroids = [np.mean(data[cluster_assgn == k], axis=0) for k in range(0,self.n_cluster)]
        pass
        
    def evaluate(self, data, cluster_assign):
        """
        K-Means Objective
        Args:
            data: Test data (M x D) matrix (numpy float)
            cluster_assign: M vector, Cluster assignment of all the samples in `data`
        Returns:
            metric : (float.)
        """
        #TODO
        total_sum = 0
        for i in range(0,len(self.centroids)):
            for clus in range(0,len(cluster_assign)):
                if(cluster_assign[clus]==i):
                    total_sum = total_sum + np.square(dist(data[clus], self.centroids[i]))
        return total_sum
