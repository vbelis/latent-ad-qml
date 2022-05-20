import numpy as np
import time
import distance_calc as distc

def initialize_centroids(points, k):
    """
    Randomly initialize centroids for cluster.
    Args:
        points: numpy.ndarray of shape (N, X)
                    N = number of events,
                    X = dimension of latent space - number of features
        k: int - number of clusters
    """
    indexes = np.random.randint(points.shape[0], size=k)
    return points[indexes]

def find_distance_matrix_quantum(XA, XB):
    
    
    XA = np.asarray(XA)
    XB = np.asarray(XB)

    sA = XA.shape
    sB = XB.shape
    
    if len(sA) != 2:
        raise ValueError('XA must have 2 dimensions.')
    if len(sB) != 2:
        raise ValueError('XB must have 2 dimensions.')
    if sA[1] != sB[1]:
        raise ValueError('XA and XB must have the same feature dimension.')

    mA = sA[0]; mB = sB[0]; n = sA[1]
    dist_matrix = np.zeros((mA, mB))
    for i in range(mA):
        distance, _ = distc.DistCalc_DI(XA[i,:], XB[0])
        dist_matrix[i,:] = distance
    return dist_matrix

def geometric_median(X, eps=1e-6):
    if X.size==0: 
        print("For this class there is no points assigned!")
        return
    y = np.mean(X, 0)
    #print(f'First median is: {y}')
    z=0
    while True:
        D = find_distance_matrix_quantum(X, [y])
        nonzeros = (D != 0)[:, 0] #which points are not equal to y
        Dinv = 1 / D[nonzeros]
        Dinv_sum = np.sum(Dinv)
        W = Dinv / Dinv_sum
        T1 = np.sum(W * X[nonzeros], 0) #scaled sum of all points
        
        num_zeros = len(X) - np.sum(nonzeros) # number of points = y
        #print(f'Size of inputs: {X.shape}, and there is 0: {num_zeros}')
        if num_zeros == 0: #then next median is scaled sum of all points
            y1 = T1
        elif num_zeros == len(X):
            return y
        else:
            R = (T1 - y) * Dinv_sum
            r = np.linalg.norm(R)
            gamma = 0 if r == 0 else num_zeros/r
            gamma = min(1, gamma)
            y1 = (1-gamma)*T1 + gamma*y
        
        # converge condition    
        dist_y_y1,_ = distc.DistCalc_DI(y, y1)
        #print(f'Distance between 2 medians: {dist_y_y1}')
        if dist_y_y1 < eps:
            return y1
        #print(f'Next median is: {y1}')
        y = y1 # next median is
        z+=1
        
def find_centroids_GM(points, cluster_labels, clusters=2):
    start_time = time.time()
    
    centroids = np.zeros([clusters, points.shape[1]])
    k = points.shape[1]
    for j in range(clusters):
        print(f'Searching centroids for cluster {j}')
        points_class_i = points[cluster_labels==j]
        median = geometric_median(points_class_i)
        centroids[j,:] = median
        print(f'Found for cluster {j}')
    print("MedianCalc ---> %s seconds ---" % (time.time() - start_time))
    return np.array(centroids)

def find_nearest_neighbour_DI(points, centroids):
    start_time = time.time()
    """
    Find nearest neighbours using destructive inference probabilities.
    Args:
        points: numpy.ndarray of shape (N, X)
                    N = number of samples,
                    X = dimension of latent space;
        centroids: numpy.ndarray of shape (N, X)
    Returns:
        cluster_assignments: numpy.ndarray of shape (N, X) specifying to which cluster each feature is assigned
        distances: numpy.ndarray of shape (N, X) specifying distances to nearest cluster
    """
    
    n = points.shape[0]
    num_features = points.shape[1]
    k = centroids.shape[0] # number of centroids
    #cluster_label = np.zeros(n) # assignment to new centroids
    cluster_label=[]
    distances=[]
    
    for i in range(n): # through all training samples
        dist=[]
        #print(f'Point for distance: {points[i,:]}')
        for j in range(k): # distance of each training example to each centroid
            #print(f'Centroid for distance: {centroids[j,:]}')
            temp_dist, _ = distc.DistCalc_DI(points[i,:], centroids[j,:], shots_n=10000) # returning back one number for all latent dimensions!
            dist.append(temp_dist)
        #cluster_index = m.duerr_hoyer_algo(dist)
        cluster_index = np.argmin(dist)
        cluster_label.append(cluster_index)
        distances.append(dist)
    print("Find Cluster Labels ---> %s seconds ---" % (time.time() - start_time))
    return np.asarray(cluster_label), np.asarray(distances)