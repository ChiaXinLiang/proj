import cv2
import sys
import numpy as np
import time
import math

class SLIC:
    def __init__(self, img, step, nc):
        self.img = img
        self.greyImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.height, self.width = img.shape[:2]
        self._convertToLAB()
        self.step = step
        #maximum color distance
        self.nc = nc
        #maximum spatial distance
        self.ns = step
        self.FLT_MAX = 1000000
        self.ITERATIONS = 10

    def _convertToLAB(self):
        try:
            import cv2
            self.labimg = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB).astype(np.float64)
        except ImportError:
            self.labimg = np.copy(self.img)
            #Y.shape is (n,m), so Y.shape[0] means number of row
            for i in range(self.labimg.shape[0]):
                for j in range(self.labimg.shape[1]):
                    rgb = self.labimg[i, j]
                    self.labimg[i, j] = self._rgb2lab(tuple(reversed(rgb)))

    def _rgb2lab ( self, inputColor ) :

       num = 0
       RGB = [0, 0, 0]

       for value in inputColor :
           value = float(value) / 255

           if value > 0.04045 :
               value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
           else :
               value = value / 12.92

           RGB[num] = value * 100
           num = num + 1

       XYZ = [0, 0, 0,]

       X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
       Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
       Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
       XYZ[ 0 ] = round( X, 4 )
       XYZ[ 1 ] = round( Y, 4 )
       XYZ[ 2 ] = round( Z, 4 )

       XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
       XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
       XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

       num = 0
       for value in XYZ :

           if value > 0.008856 :
               value = value ** ( 0.3333333333333333 )
           else :
               value = ( 7.787 * value ) + ( 16 / 116 )

           XYZ[num] = value
           num = num + 1

       Lab = [0, 0, 0]

       L = ( 116 * XYZ[ 1 ] ) - 16
       a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
       b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

       Lab [ 0 ] = round( L, 4 )
       Lab [ 1 ] = round( a, 4 )
       Lab [ 2 ] = round( b, 4 )

       return Lab

    def generateSuperPixels(self):
        self._initData()
        #np.mgrid returns a dense multi-dimensional "meshgrid"
        #swapaxes(axis1,axis2) interchange two axes of an arrays
        indnp = np.mgrid[0:self.height,0:self.width].swapaxes(0,2).swapaxes(0,1)
        for i in range(self.ITERATIONS):
            self.distances = self.FLT_MAX * np.ones(self.img.shape[:2])
            for j in range(self.centers.shape[0]):
                xlow, xhigh = int(self.centers[j][3] - self.step), int(self.centers[j][3] + self.step)
                ylow, yhigh = int(self.centers[j][4] - self.step), int(self.centers[j][4] + self.step)

                if xlow <= 0:
                    xlow = 0
                if xhigh > self.width:
                    xhigh = self.width
                if ylow <=0:
                    ylow = 0
                if yhigh > self.height:
                    yhigh = self.height

                cropimg = self.labimg[ylow : yhigh , xlow : xhigh]  
                colordiff = cropimg - self.labimg[int(self.centers[j][4]), int(self.centers[j][3])]
                colorDist = np.sqrt(np.sum(np.square(colordiff), axis=2))

                yy, xx = np.ogrid[ylow : yhigh, xlow : xhigh]
                pixdist = ((yy-self.centers[j][4])**2 + (xx-self.centers[j][3])**2)**0.5
                dist = ((colorDist/self.nc)**2 + (pixdist/self.ns)**2)**0.5

                distanceCrop = self.distances[ylow : yhigh, xlow : xhigh]
                idx = dist < distanceCrop
                distanceCrop[idx] = dist[idx]
                self.distances[ylow : yhigh, xlow : xhigh] = distanceCrop
                #wtf is this since cluster is 2D matrix why can add [idx]
                self.clusters[ylow : yhigh, xlow : xhigh][idx] = j


            for k in range(len(self.centers)):
                idx = (self.clusters == k)
                colornp = self.labimg[idx]
                distnp = indnp[idx]
                self.centers[k][0:3] = np.sum(colornp, axis=0)
                sumy, sumx = np.sum(distnp, axis=0)
                self.centers[k][3:] = sumx, sumy
                self.centers[k] /= np.sum(idx)


    def _initData(self):
        #img.shape[:2] only show the height and width of image
        self.clusters = -1 * np.ones(self.img.shape[:2])
        self.distances = self.FLT_MAX * np.ones(self.img.shape[:2])

        centers = []
        for i in range(self.step, self.width - int(self.step/2), self.step):
            for j in range(self.step, self.height - int(self.step/2), self.step):
                
                nc = self._findLocalMinimum(center=(i, j))
                color = self.labimg[nc[1], nc[0]]
                center = [color[0], color[1], color[2], nc[0], nc[1]]
                centers.append(center)
        self.center_counts = np.zeros(len(centers))
        self.centers = np.array(centers)


    def displayContours(self, color):
        dx8 = [-1, -1, 0, 1, 1, 1, 0, -1]
        dy8 = [0, -1, -1, -1, 0, 1, 1, 1]

        isTaken = np.zeros(self.img.shape[:2], np.bool)
        contours = []

        for i in range(self.width):
            for j in range(self.height):
                nr_p = 0
                for dx, dy in zip(dx8, dy8):
                    x = i + dx
                    y = j + dy
                    if x>=0 and x < self.width and y>=0 and y < self.height:
                        if isTaken[y, x] == False and self.clusters[j, i] != self.clusters[y, x]:
                            nr_p += 1

                if nr_p >= 2:
                    isTaken[j, i] = True
                    contours.append([j, i])

        for i in range(len(contours)):
            self.img[contours[i][0], contours[i][1]] = color

    def _findLocalMinimum(self, center):
        min_grad = self.FLT_MAX
        loc_min = center
        for i in range(center[0] - 1, center[0] + 2):
            for j in range(center[1] - 1, center[1] + 2):
                c1 = self.labimg[j+1, i]
                c2 = self.labimg[j, i+1]
                c3 = self.labimg[j, i]
                if ((c1[0] - c3[0])**2)**0.5 + ((c2[0] - c3[0])**2)**0.5 < min_grad:
                    min_grad = abs(c1[0] - c3[0]) + abs(c2[0] - c3[0])
                    loc_min = [i, j]
        return loc_min

    def MakeSmallImage(self):
        n,m = self.img.shape[0:2] 

        clusterNumber = self.clusters.reshape(n*m,1)
        maxCluster = max(clusterNumber)
        allpixel = []
    
        for j in range(1,int(maxCluster)+1):
            superPixelPiece = 255*np.ones((n,m,3), np.int8)
            superPixelPiece1 = 255*np.ones((2*self.step,2*self.step,3), np.int8) 
            for i in range(0,n*m):
                if clusterNumber[i] == j:
                    superPixelPiece[math.floor(i/m),int(i%m)] = self.img[math.floor(i/m),int(i%m)]

            centers = self.centers[j]
            xlow = int(centers[4]) - self.step 
            xhigh = int(centers[4]) + self.step
            ylow = int(centers[3]) - self.step
            yhigh = int(centers[3]) + self.step

            if xlow <= 0:
                xlow = 0
            if xhigh > self.width:
                xhigh = self.width
            if ylow <=0:
                ylow = 0
            if yhigh > self.height:
                yhigh = self.height

            superPixelPiece1[0:(xhigh-xlow),0:(yhigh-ylow)]= superPixelPiece[xlow:xhigh, ylow:yhigh]
            maxRow, maxCol = superPixelPiece1.shape[0:2]
            if maxRow < 2*self.step:
                superPixelPiece1[maxRow:(2*self.step+1),:]=[255,255,255]
            if maxCol < 2*self.step:
                superPixelPiece1[:,maxCol:(2*self.step+1)] =[255,255,255]
            allpixel.append(superPixelPiece1)
        
        return maxCluster, allpixel

    def MakeSmallGreyImg(self):
        n,m = self.greyImg.shape[0:2] 

        clusterNumber = self.clusters.reshape(n*m,1)
        maxCluster = max(clusterNumber) 
        allpixel = []
    
        for j in range(1,int(maxCluster)+1):
            superPixelPiece1 = 255*np.ones((2*self.step,2*self.step,3), np.int8)
            superPixelPiece = 255*np.ones((n,m,3), np.int8)
            for i in range(0,n*m):
                if clusterNumber[i] == j:
                    superPixelPiece[math.floor(i/m),int(i%m)] = self.greyImg[math.floor(i/m),int(i%m)]

            centers = self.centers[j]
            xlow = int(centers[4]) - self.step 
            xhigh = int(centers[4]) + self.step
            ylow = int(centers[3]) - self.step
            yhigh = int(centers[3]) + self.step

            if xlow <= 0:
                xlow = 0
            if xhigh > self.width:
                xhigh = self.width
            if ylow <=0:
                ylow = 0
            if yhigh > self.height:
                yhigh = self.height

            superPixelPiece1[0:(xhigh-xlow),0:(yhigh-ylow)]= superPixelPiece[xlow:xhigh, ylow:yhigh]
            maxRow, maxCol = superPixelPiece1.shape[0:2]
            if maxRow < 2*self.step:
                superPixelPiece1[maxRow:(2*self.step+1),:]=[255,255,255]
            if maxCol < 2*self.step:
                superPixelPiece1[:,maxCol:(2*self.step+1)] =[255,255,255]
            allpixel.append(superPixelPiece1)
        
        return allpixel
