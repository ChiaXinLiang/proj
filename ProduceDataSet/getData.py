from feaAlg import *
import os


"""
get feature
"""
def getFea(img):
    refImg = cv2.resize(img,(500,500))
    nr_superpixels = int(150)
    nc = int(100)
    step = int((refImg.shape[0]*refImg.shape[1]/nr_superpixels)**0.5)
    refSLIC = SLIC(refImg,step,nc)
    refSLIC.generateSuperPixels()
#    refSLIC.displayContours((0,0,0))
#    cv2.imwrite("img3.jpg",refSLIC.img)

    #get the piece of super pixel
    maxCluster, AllRefPixel = refSLIC.MakeSmallImage()
    AllTarPixel = refSLIC.MakeSmallGreyImg()

    return maxCluster, AllRefPixel, AllTarPixel


"""
Get the list of Images
"""
def get_imlist(path):
        
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')] 



"""
Produce the Data Set
"""
def getDataSet(path):
    imlist = get_imlist(path)
    numData = len(imlist)

    DataRefSet = []
    DataTarSet = []
    for i in range(0,numData):
        Img = cv2.imread(imlist[i])
        maxCluster, AllRefPixel, AllTarPixel = getFea(Img)
        DataRefSet.append(AllRefPixel)
        DataTarSet.append(AllTarPixel)

    return DataRefSet, DataTarSet





