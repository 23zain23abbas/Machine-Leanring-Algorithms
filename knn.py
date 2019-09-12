from __future__ import division, print_function
from collections import Counter
from typing import List

import numpy as np
import scipy


class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        self.features = features
        self.labels = labels
        return
        

    #TODO: predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        predictedLabels = [-1] * len(features)

        for i in range(len(features)):
            point = features[i]
            labels = self.get_k_neighbors(point)
            labelCounter = Counter(labels)
            maxLabel = labelCounter.most_common(1)[0][0]
            predictedLabels[i] = maxLabel
            
            
        return predictedLabels

    #TODO: find KNN of one point
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        neighbors = []
        for index in range(len(self.features)):
            distance = self.distance_function(point, self.features[index])
            neighbors.append([distance, self.labels[index]])
        
        neighbors.sort(key=lambda x: x[0])
        return [i[1] for i in neighbors[:self.k]]
        

if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
