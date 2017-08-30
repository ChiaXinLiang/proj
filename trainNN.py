from scipy import io
import cv2

#Change all image into 500*500
#number of cluster = 120
#dimenstion of Superpixel = 80*80

dataSet = io.loadmat('ProduceDataSet/Data.mat')
DataColorSet = dataSet['Color']
DataGreySet = dataSet['Grey']
n = DataColorSet.shape[0]
print(n)

cv2.imwrite("img1.jpg",DataColorSet[0][1])
cv2.imwrite("img2.jpg",DataColorSet[3][56])
cv2.imwrite("img3.jpg",DataGreySet[0][1])
cv2.imwrite("img4.jpg",DataGreySet[3][56])

