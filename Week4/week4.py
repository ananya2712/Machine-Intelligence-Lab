import numpy as np
from collections import defaultdict


def minkowski_distance(a, b, p):
  sum=0
  length = len(a)
  for i in range(length):
    sum = sum + (abs(a[i]-b[i]))**p
  sum = sum**(1/p)
  return sum
class KNN:

    def _init_(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        self.data = data
        self.target = target.astype(np.int64)

        return self

    def find_distance(self, x):
        dist = []
        for vector in x:
            dist.append((np.abs(vector - self.data)*self.p).sum(axis=1)* (1 / self.p))
        return dist 


    def k_neighbours(self, x):
        dist = self.find_distance(x)

        sort_dist = np.argsort(dist, axis=1)

        idx_of_neigh = []
        neigh_dists = []
        for i in range(len(sort_dist)):
            ndst= []
            nid = []
            for j in range(self.k_neigh):
                ndst.append(dist[i][sort_dist[i][j]])
                nid.append(sort_dist[i][j])
            neigh_dists.append(ndst)
            idx_of_neigh.append(nid)
        knn_list = [[],[]]
        knn_list[0].extend(neigh_dists)
        knn_list[1].extend(idx_of_neigh)
        return knn_list


    def predict(self, x):
        pred_val = []
        neigh_dists, idx_of_neigh = self.k_neighbours(x)
        for i in range(len(neigh_dists)):
            freq = defaultdict(lambda: 0.)
            for j in range(len(neigh_dists[i])):
                if self.weighted == True:
                    freq[self.target[idx_of_neigh[i][j]]] += 1/(neigh_dists[i][j]+1e-9)

                else:
                    freq[self.target[idx_of_neigh[i][j]]] += 1

            freq_l = list(freq.items())
            freq_l.sort(key = lambda x: x[1], reverse = True)


            pred_val.append(freq_l[0][0])

        return pred_val
        pass

    def evaluate(self, x, y):

        c=0
        pred=self.predict(x)
        n = len(pred)
        for i in range(n):
          if (pred[i]==y[i]):
            c= c + 1
        acc = c / (len(x))
        return acc * 100
