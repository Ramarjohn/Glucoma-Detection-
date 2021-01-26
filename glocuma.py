from scipy import signal
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#----------------------------PRE-PROCESSING AND SMOOTHING---------------------------------

img=mpimg.imread('D:/7sem/a.png')
Abo=img[:,:,0]
Ago=img[:,:,1]
Aro=img[:,:,2]
filter = signal.gaussian(60, std=6) #Gaussian Window
filter=filter/sum(filter)
STDf=filter.std()


Ar = Aro - Aro.mean()           #Preprocessing Red (Aro)
Ar = Ar - Ar.mean() - Aro.std() #Preprocessing Red
Ar = Ar - Ar.mean() - Aro.std() #Preprocessing Red

Mr = Ar.mean()                           #Mean of preprocessed red
SDr = Ar.std()                           #SD of preprocessed red
#Thr = 49.5 - 12 -2* Ar.std()               #OD Threshold
#Thr = Ar.std()
#Thr=Mr+2*SDr
Thr=0.4-STDf-Ar.std()
print("OD Threshold:",Thr)

#Ag = Ago - Ago.mean()           #Preprocessing Green
#Ag = Ag - Ag.mean() - Ago.std() #Preprocessing Green
Ag=Ago-Ago.mean()-Ago.std()




hist,bins = np.histogram(Ag.ravel(),256,[0,256])   #Histogram of preprocessed green channel
histr,binsr = np.histogram(Ar.ravel(),256,[0,256]) #Histogram of preprocessed red channel

smooth_hist_g=np.convolve(filter,hist)  #Histogram Smoothing Green
smooth_hist_r=np.convolve(filter,histr) #Histogram Smoothing Red

#plt.subplot(2, 2, 1)
#plt.plot(hist)
#plt.title("Preprocessed Green Channel")

#plt.subplot(2, 2, 2)
#plt.plot(smooth_hist_g)
#plt.title("Smoothed Histogram Green Channel")

#plt.subplot(2, 2, 3)
#plt.plot(histr)
#plt.title("Preprocessed Red Channel")

#plt.subplot(2, 2, 4)
#plt.plot(smooth_hist_r)
#plt.title("Smoothed Histogram Red Channel")

#plt.show()

#---------------------------------APPLYING THRESHOLD--------------------------------------

r,c = Ag.shape
Dd = np.zeros(shape=(r,c))
Dc = np.zeros(shape=(r,c))

for i in range(1,r):
	for j in range(1,c):
		if Ar[i,j]>Thr:
			Dd[i,j]=255
		else:
			Dd[i,j]=90

for i in range(1,r):
	for j in range(1,c):
		if Ag[i,j]>Thg:
			Dc[i,j]=1
		else:
			Dc[i,j]=0
		
#----------SELECTING LARGEST AREA AND PERFORMING MORPHOLOGY TO FIND EXACT CUP-------------
#L1=cv2.connectedComponents(Dd,


#------------------------DISPLAYING SEGMENTED OPTIC DISK AND CUP--------------------------
imgplot = plt.imshow(img)
plt.show()
imgplot = plt.imshow(Abo)
plt.show()
imgplot = plt.imshow(Ago)
plt.show()
imgplot = plt.imshow(Aro)
plt.show()
imgplot=plt.imshow(Dd, cmap = 'gray', interpolation = 'bicubic')
plt.axis("off")
plt.title("Optic Disk")
plt.show()

#imgplot=plt.imshow(Dc, cmap = 'gray', interpolation = 'bicubic')
#plt.axis("off")
#plt.title("Optic Cup")
#plt.show()
#c=Thg/Thr

