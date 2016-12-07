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
	newimage = cv2.resize(image, (1280, 967), interpolation=cv2.INTER_AREA)
	cv2.imwrite('1280_967.bmp', newimage)
	return newimage


def change_to_gray(image):
	'''
	Change a RGB picture to a Gray picture
	:param image: narray, 3-dimension picture
	:return: narray, 2-dimension, gray picture
	'''
	myimg1 = image.copy()
	gray_img = cv2.cvtColor(myimg1, cv2.COLOR_BGR2GRAY)
	cv2.imwrite("gray.bmp", gray_img)
	return gray_img


def cut(img, mode=1):
	'''
	Cut a picture to get its upper part or lower part
	:param img: narray
	:param mode: int, '1' for upper part, '2' for lower part, Default:'1'
	:return: narray, picture after cutting
	'''
	h = img.shape[0]
	w = img.shape[1]
	if mode == 1:
		cutimg = img[int(h / 2.2):int(h / 1.7), int(w / 3.8):int(w / 1.8) + 15]  # up
	else:
		cutimg = img[int(h / 1.7):int(h / 1.4) + 7, int(w / 3.8):int(w / 1.8) + 15]  # down
	cv2.imwrite("after_cut.bmp", cutimg)
	return cutimg


def pre_process(filename, mode=1):
	'''

	:param filename: string, name of the picture, eg: 'test.bmp'
	:param mode: int, '1' for upper part, '2' for lower part, Default:'1'
	:return: narray, picture after pre-processing
	'''

	newimage = change_size(filename)
	gray_img = change_to_gray(newimage)
	cutimg = cut(gray_img,mode)
	return cutimg

