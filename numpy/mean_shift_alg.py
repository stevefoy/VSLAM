import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth

#Loading original image



def test(originImg):
    # Shape of original image
    
    originShape = originImg.shape
    
    
    # Converting image into array of dimension [nb of pixels in originImage, 3]
    
    # based on r g b intensities
    
    flatImg=np.reshape(originImg, [-1, 3])
    
    
    # Estimate bandwidth for meanshift algorithm
    
    bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)
    
    ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)
    
    
    # Performing meanshift on flatImg
    
    ms.fit(flatImg)
    
    
    # (r,g,b) vectors corresponding to the different clusters after meanshift
    
    labels=ms.labels_
    
    
    # Remaining colors after meanshift
    
    cluster_centers = ms.cluster_centers_
     
    
    # Finding and diplaying the number of clusters
    
    labels_unique = np.unique(labels)
    
    n_clusters_ = len(labels_unique)
    
    print("number of estimated clusters : %d" % n_clusters_)
    
    
    # Displaying segmented image
    segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]
    
    return segmentedImg

originImg = cv2.imread('/home/stephen/Pictures/Road_Data_Valeo/smallTShapes/FV/0000510562.png')
originImg2 = cv2.imread('/home/stephen/Pictures/Road_Data_Valeo/smallTShapes/FV/0000512262.png')

segmentedImg1 = test(originImg)
segmentedImg2 = test(originImg2)

cv2.imshow('Image',segmentedImg1.astype(np.uint8))
cv2.imshow('Image2',segmentedImg2.astype(np.uint8))
cv2.waitKey(0)

cv2.destroyAllWindows()

