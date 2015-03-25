import cv2
import numpy as np
import data
import shutil,sys
import skimage.feature
from sklearn import preprocessing as prep
from sklearn.decomposition import PCA
from skimage import measure
from skimage import morphology
from skimage.feature import peak_local_max
from skimage import segmentation
from skimage.morphology import watershed

import mahotas as mh

if not len(sys.argv) == 2: 
    subset = "train"
    print "Usage extract_features.py <subset:train or test (default=train)>"
else:
    subset = sys.argv[1]

SAVE_PATH = "data/features_%s.pkl"%subset
scale = True

images = data.load(subset)
# images_test = data.load('test')

print images.shape

def pause(): 
    import time
    time.sleep(10000)

def getLargestRegion(props, labelmap, imagethres):
    regionmaxprop = None
    for regionprop in props:
        # check to see if the region is at least 50% nonzero
        if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop

def getMinorMajorRatio_2(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)
 
    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))
 
    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)
        
    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    minor_axis_length = 0.0
    major_axis_length = 0.0
    area = 0.0
    convex_area = 0.0
    eccentricity = 0.0
    equivalent_diameter = 0.0
    euler_number = 0.0
    extent = 0.0
    filled_area = 0.0
    orientation = 0.0
    perimeter = 0.0
    solidity = 0.0
    centroid = [0.0,0.0]
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
        minor_axis_length = 0.0 if maxregion is None else maxregion.minor_axis_length 
        major_axis_length = 0.0 if maxregion is None else maxregion.major_axis_length  
        area = 0.0 if maxregion is None else maxregion.area  
        convex_area = 0.0 if maxregion is None else maxregion.convex_area  
        eccentricity = 0.0 if maxregion is None else maxregion.eccentricity  
        equivalent_diameter = 0.0 if maxregion is None else maxregion.equivalent_diameter  
        euler_number = 0.0 if maxregion is None else maxregion.euler_number  
        extent = 0.0 if maxregion is None else maxregion.extent 
        filled_area = 0.0 if maxregion is None else maxregion.filled_area  
        orientation = 0.0 if maxregion is None else maxregion.orientation 
        perimeter = 0.0 if maxregion is None else maxregion.perimeter  
        solidity = 0.0 if maxregion is None else maxregion.solidity
        centroid = [0.0,0.0] if maxregion is None else maxregion.centroid
 
    return ratio,minor_axis_length,major_axis_length,area,convex_area,eccentricity,\
           equivalent_diameter,euler_number,extent,filled_area,orientation,perimeter,solidity, centroid[0], centroid[1]

features = {"hu":[],"ORB":[],"tutorial":[],"haralick":[],"lbp":[],"pftas":[],
    "zernike_moments":[],"image_size":[]}

image_shapes = np.asarray([img.shape for img in data.load('train')]).astype(np.float32)
# print image_shapes.shape
# pause()

# img_shp = (64,64)
# resized_imgs = np.empty((len(images),np.prod(img_shp)),dtype="uint8")
report = [int((j+1)*images.shape[0]/100.) for j in range(100)]
# count = 0
for i,img in enumerate(images):
    img_o = img.copy()
    img = data.uint_to_float(img)

    # Hu moments
    hu = cv2.HuMoments(cv2.moments(img)).flatten()
    features["hu"].append(hu)

    # ORB
    # img_ = cv2.resize(img_o,img_shp)
    # orb = cv2.ORB(nfeatures=1,patchSize=5,nlevels=4,edgeThreshold=5)
    # kp, des = orb.detectAndCompute(img_,None)
    # if (des is None):
    #     des = np.zeros((32,))
    #     count+=1
    # else:
    #     des = des.flatten()
    # features["ORB"].append(des)

    #Tutorial features
    tut_features = np.array(getMinorMajorRatio_2(img_o))

    # image2 = mh.imread(nameFileImage, as_grey=True)
    haralick = mh.features.haralick(img_o, ignore_zeros=False, 
        preserve_haralick_bug=False, compute_14th_feature=False).flatten()
    lbp      =  mh.features.lbp(img_o, radius=20, points=7, ignore_zeros=False)
    pftas    = mh.features.pftas(img_o)
    zernike_moments = mh.features.zernike_moments(img_o, radius=20, degree=8)

    # print tut_features.shape, tut_features.dtype
    # print haralick.shape, haralick.dtype
    # print lbp.shape, lbp.dtype
    # print pftas.shape, pftas.dtype
    # print zernike_moments.shape, zernike_moments.dtype
    # print np.array(img_o.shape).shape

    features["tutorial"].append(tut_features)
    features["haralick"].append(haralick)
    features["lbp"].append(lbp)
    features["pftas"].append(pftas)
    features["zernike_moments"].append(zernike_moments)
    features["image_size"].append(np.array(img_o.shape))

    # pause()
    
    if i in report: print np.ceil(i *100.0 / images.shape[0]), "% done"

    # resized_imgs[i] = img_.flatten()



# print count
# print "PCA"
# pca = PCA(n_components=10)
# pca.fit(resized_imgs)
# x_pca = pca.transform(resized_imgs)
# # print x_pca[0]
# # pause()
# if scale:
#     scaler = prep.StandardScaler().fit(x_pca)
#     x_pca = scaler.transform(x_pca)


for k,v in features.items(): 
    x = np.array(v,dtype="float32")
    if scale:
        scaler = prep.StandardScaler().fit(x)
        x = scaler.transform(x)
    features.update({k:x})

# features["PCA"] = x_pca

np.save(open(SAVE_PATH,"wb"),features)
# shutil.move("/home/lpigou/temp.np",SAVE_PATH)
