# -*- coding: utf8 -*-
__author__ = 'PowerDi'
__address__ = '578661971@qq.com'
import numpy as np
import milk
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pre_processing
import time
from skimage import exposure
from skimage.feature import greycomatrix,greycoprops
from skimage.measure import find_contours
from skimage.measure import label,regionprops
from skimage.morphology import remove_small_holes,remove_small_objects
from skimage.morphology import closing,dilation,erosion
from skimage.morphology import square,rectangle

def segmentation(copyimg, SIZE_h=10, SIZE_w=30):
	'''
	Divide a picture to several blocks. This situation generates
	rectangle block.
	:param image:Narray,Input image
	:param SIZE_h:int, height of the small rectangle block. Default:10
	:param SIZE_w:int, width of the small rectangle block. Default:30

	'''
	h, w = copyimg.shape
	patch = []
	cpatch = []
	# segmetation
	for j in range(0, h, 3):
		for i in range(0, w, 20):
			if j+SIZE_h <= h:
				if i + SIZE_w <= w:
					patch.append(copyimg[j:(j + SIZE_h), i:(i + SIZE_w)])
					cpatch.append(copyimg[j:j + (SIZE_h), i:(i + SIZE_w)])
				else:
					patch.append(copyimg[j:(j + SIZE_h), i:])
					cpatch.append(copyimg[j:(j + SIZE_h), i:])
			else:
				if i + SIZE_w <= w:
					patch.append(copyimg[j:, i:(i + SIZE_w)])
					cpatch.append(copyimg[j:, i:(i + SIZE_w)])
				else:
					patch.append(copyimg[j:, i:])
					cpatch.append(copyimg[j:, i:])
	if h % 3 == 0:
		num_h = h / 3
	else:
		num_h = h / 3 + 1
	if w % 20 == 0:
		num_w = w / 20
	else:
		num_w = w / 20 + 1
	return patch, cpatch, num_h, num_w

def init_entropy(block):
	'''
	Calculate the entropy of a block
	:param block: narray, input sequence.
	:return: float, the entropy of the input sequence
	'''
	assert block.ndim == 2, 'must input a gray image'
	block_his = exposure.histogram(block, nbins=256)[0]
	block_his = [x for x in block_his if x > 0]
	block_his = np.array(block_his)
	ent = -1 * block_his * np.log2(block_his)
	ent2 = np.sum(ent)
	return ent2


def weld_dectection(patch, num_h, num_w):
	'''
	weld dectection algorithm
	:param patch:
	:param num_h:
	:param num_w:
	:return: labered array
	'''
	xs = []
	for block in patch:
		glcm = greycomatrix(block, [2], [np.pi / 4], 256,
		                    symmetric=True, normed=False)
		ent = init_entropy(block.copy())
		con = greycoprops(glcm, 'contrast')[0, 0]
		xs.append([con, ent])
	# K-means
	xs = np.array(xs)
	assignments, centroids = milk.kmeans(xs, 2)
	# fix the label of weld to be number 1
	numones = np.sum(assignments == 1)
	numzeros = np.sum(assignments == 0)
	numtwos = np.sum(assignments == 2)
	if max(numones, numzeros, numtwos) == numzeros:
		tip = [0 if ax == 0 else 1 for ax in assignments]
	elif max(numones, numzeros, numtwos) == numones:
		tip = [0 if ax == 1 else 1 for ax in assignments]
	elif max(numones, numzeros, numtwos) == numtwos:
		tip = [0 if ax == 2 else 1 for ax in assignments]
	assignments = np.array(tip)
	assign = assignments.reshape(num_h, num_w)
	return assign

def label_image(bw):
	label_image = label(bw, connectivity=1)
	r = []
	c = []
	for region in regionprops(label_image):
		if region.area >= 720:
			minr, minc, maxr, maxc = region.bbox
			if minr < 63 and maxr > 63:
				r.append(minr)
				r.append(maxr)
				c.append(minc)
				c.append(maxc)
				minr = min(r)
				maxr = max(r)
				minc = min(c)
				maxc = max(c)
	rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
	ax.add_patch(rect)
	return minr, maxr, minc, maxc

def OTSU_enhance(img_gray, th_begin=0, th_end=256, th_step=1):
    assert img_gray.ndim == 2, "must input a gary_img"

    max_g = 0
    suitable_th = 0
    for threshold in xrange(th_begin, th_end, th_step):
        bin_img = img_gray > threshold
        bin_img_inv = img_gray <= threshold
        fore_pix = np.sum(bin_img)
        back_pix = np.sum(bin_img_inv)
        if 0 == fore_pix:
            break
        if 0 == back_pix:
            continue

        w0 = float(fore_pix) / img_gray.size
        u0 = float(np.sum(img_gray * bin_img)) / fore_pix
        w1 = float(back_pix) / img_gray.size
        u1 = float(np.sum(img_gray * bin_img_inv)) / back_pix
        # intra-class variance
        g =  w0 * w1  *(u0 - u1) * (u0 - u1)
        if g > max_g:
            max_g = g
            suitable_th = threshold*w0
    return suitable_th

def calculate_area(contour):
    AreaSum = 0.0
    for i in range(0, len(contour)-1):
        x, x1 = contour[i], contour[i+1] #  Area=\sum_{n=1}^N(X_(i+1)*Y_i+X_i*Y_(i+1))
        AreaSum = AreaSum+(x1[1]*x[0]-x[1]*x1[0])
    dArea = abs(AreaSum/2.0)
    return dArea

#*********************************************#
filename = 'test.bmp'
mode = 1 # 1 for upper part, 2 for lower part
#*********************************************#

image = pre_processing.pre_process(filename, mode)
h, w = image.shape
copyimg = image.copy() # may have wrong
patch, cpatch, num_h, num_w = segmentation(copyimg, SIZE_h=10,
                                           SIZE_w=20)
assign = weld_dectection(patch, num_h, num_w)


# morphology process
element = rectangle(2, 1)
element2 = rectangle(1, 1)
assign_2 = erosion(assign, selem=element)
assign_2 = dilation(assign_2, selem=element2)
assign_2 = remove_small_objects(assign_2.astype(bool), 20).astype(int)
assign_2 = remove_small_holes(assign_2.astype(bool), 30).astype(int)
assign_2 = assign_2.ravel()


# change picture into binary picture
count = 0
for block2 in cpatch:
	if assign_2[count] == 0:
		block2[:] = 0
	else:
		block2[:] = 255
	count+=1
copyimg[:, w-10:] = 0
element3 = rectangle(7, 1)
bw = erosion(copyimg, selem=element3)
bw = dilation(bw, selem=element3)

# clear borders if the sum of pixel along y axis is under 9
answer = np.sum(bw == 255, axis=0)
for bb in range(0, w):
    if answer[bb] < 9:
        bw[:, bb] = 0
    else:
        pass

# draw the result
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(2, 1, 1)
ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest',
          vmin=0, vmax=255)
ax.axis('off')
fig.suptitle('result', fontsize=14)
minr, maxr, minc, maxc = label_image(bw)

# otsu
newimage = image[minr:maxr, minc:maxc]
thres  =OTSU_enhance(newimage)
bw2 = newimage < thres
bw2 = remove_small_objects(bw2.astype(bool), 50).astype(int)
bw2 = closing(bw2, square(5))

# draw the contours
contours=find_contours(bw2,0.5)
ax1=fig.add_subplot(2,1,2)#add_subplot返回的是子图名称
ax1.imshow(newimage,cmap=plt.cm.gray,interpolation='nearest')
ax1.axis('off')
for n,contour in enumerate(contours):
    print n,calculate_area(contour)
    if calculate_area(contour)>50:
        ax1.plot(contour[:,1],contour[:,0],linewidth=2)

fig.tight_layout()
plt.show()