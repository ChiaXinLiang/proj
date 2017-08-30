from getData import *
from scipy import io

#Change all image into 500*500
#number of cluster = 120
#dimenstion of Superpixel = 80*80
path="Images"
DataColorSet, DataGreySet = getDataSet(path)
data={'Color':DataColorSet, 'Grey':DataGreySet}
io.savemat('Data.mat',data)


