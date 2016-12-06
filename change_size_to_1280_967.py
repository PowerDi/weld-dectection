# -*- coding: utf8 -*-
__author__ = 'PowerDi'
__address__ = '578661971@qq.com'

import cv2
import numpy as np

def change_size(filename):

	'''change the initial picture to the size 1280*967
	Need openCV and numpy module(cv2)
	:param filename: eg: test.bmp
	:return: None
	'''

	fn = filename
	image = cv2.imread(fn)
	newimage = cv2.resize(image, (1280,967),
	                      interpolation=cv2.INTER_AREA)
	cv2.imwrite('1280_967.bmp', newimage)

filename = 'test.bmp'
change_size(filename)
