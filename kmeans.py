import numpy as np
import time

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):

    
    first_index = generator.randint(0, len(x))
    
    # Setting all centers to first_index at start
    centers = np.zeros(n_cluster, dtype=np.int32)
    centers[:] = first_index
    
    #Obtaining positions of centroids
    centers_data = x[centers]
    
    for i in range(1, n_cluster):
        # Array to store distance to closest centroid squared, for each point
        dist_to_min_squared = np.zeros(len(x))
        
        for k in range(len(x)):
            # Get closest centroid for each point
            minidx = np.sqrt(np.sum((centers_data - x[k]) ** 2, axis=1)).argmin()
            
            # Store distance value squared
            dist_to_min_squared[k] = np.square(np.linalg.norm(x[k]-centers_data[minidx]))

        #Array to store prob of each point being selected as next centroid
        prob_of_point = np.zeros(len(x))
        
        for l in range(len(x)):
            #Calculate Prob
            prob_of_point[l] = dist_to_min_squared[l]/np.sum(dist_to_min_squared)

        # Find point with highest prob
        max_prob_idx = np.argmax(prob_of_point)

        # Update centers and centers_data with new point
        centers[i] = max_prob_idx
        centers_data[i] = x[max_prob_idx]


    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers.tolist()



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():


    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):
    
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape
            
            
        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        
        # Index of Points that are centers
        centers = np.asarray(self.centers)
        
        # Locations of points that are centers
        means = np.asarray(x[centers])
        
        # Setting J as very High
        j = 10**10
        
        # Setting index of closest centroid as 0 for all points
        y = np.zeros(N, dtype = np.int32)

        
        
        start_time = time.time()
        loop_time = time.time()    
        #For Loop max_iter times with Break
        for iterations in range(self.max_iter):
            # Stop after certain time to avoid out-of-time
            if time.time() - start_time >= 45:
                print("TimeOut: ",time.time() - start_time )
                iterations = 100
                break
                
            print("Iter", iterations,"Time", time.time() - loop_time)
            loop_time = time.time()
            
            # For each point, find closest center and assign y[i] to its index
            for i in range(N):
                y[i] = np.sqrt(np.sum((means - x[i]) ** 2, axis=1)).argmin()

                
            # Computing J using formula
            j_new = np.sqrt(np.sum((x - means[y]) ** 2, axis=1)).sum()
            if abs(j_new - j) <= self.e:
                break
            else:
                j = j_new

                
            # Updating cluster means (locations)
            for cluster_idx in range(self.n_cluster):
                # get points belonging to cluster
                points_in_centroid = np.argwhere(y == cluster_idx).transpose()[0]
                # denom = number of points in cluster
                denom = len(points_in_centroid)
                
                point_values = x[points_in_centroid]
                #num = sum of location coordinates of all
                num =  point_values.sum(axis = 0)

                # Update new mean (location) by taking average
                if denom == 0:
                    continue
                else:
                    means[cluster_idx] = num/denom


        centroids = means
        return centroids, y, iterations

        


class KMeansClassifier():


    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):


        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        
        #Getting KMeans object
        kmeans_obj = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)
        
        #Running KMeans.fit()
        # centroids are means (locations) of centers
        # y_info is the index of assigned cluster for each point
        # iterations is the iterations it took to complete
        centroids, y_info, iterations = kmeans_obj.fit(x, centroid_func)
        
        #Assigning 0 label to each centroid (for now)
        centroid_labels = np.zeros(len(centroids), dtype = np.int32)

        for i in range(len(centroids)):
            # get points belonging to cluster
            points_in_centroid = np.argwhere(y_info == i).transpose()[0]
            
            #get labels belonging to all points in cluster
            mylabels = y[points_in_centroid].transpose()
            
            # Get bincount of all unique labels
            maxval = np.bincount(mylabels)
            
            # if no labels in cluster, select 0 label, else select maxval label
            if(len(mylabels) == 0):
                centroid_labels[i] = 0
            else:
                centroid_labels[i] = maxval.argmax()
            

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):


        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape

        
        # set all default labels for points as 0
        labels = np.zeros(N, dtype = np.int32)
        
        for i in range(N):
            # Get closest center
            closest_cluster_index = np.sqrt(np.sum((self.centroids - x[i]) ** 2, axis=1)).argmin()
            
            # assign label as label of closest center
            labels[i] = self.centroid_labels[closest_cluster_index]

        return np.array(labels)
        

def transform_image(image, code_vectors):

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    new_im = image.copy()

    # Loop for all points in image
    for i in range(len(image)):
        for j in range(len(image[i])):
            
            # find code vector with closest RGB value 
            closest_vector_index = np.sqrt(np.sum((code_vectors - image[i][j]) ** 2, axis=1)).argmin()
            
            # assign RGB value of closest code_vector to image
            new_im[i][j] = code_vectors[closest_vector_index]
        

    return new_im

