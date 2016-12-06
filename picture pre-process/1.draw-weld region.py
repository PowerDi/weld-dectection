# -*- coding: utf8 -*-
__author__ = 'PowerDi'
__address__ = '578661971@qq.com'

import cv2
import numpy as np

'''
Use this tool to draw weld region. When opened a picture, using
a mouse to draw the region and the region will become blue. Press
'Esc' to quit the tool.
'''

drawing = False
mode = False
ix, iy = -1, -1

def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        print 'left button down'
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        #print 'mouse move'
        if drawing == True:
            if mode == True:
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        #print 'left button up'
        drawing = False


img = cv2.imread("test.bmp") #更换
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while (True):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xff
    if k == ord('m'):
        print 'you typed key m'
        mode = not mode
    elif k == 27:
        cv2.imwrite("afterlabel.bmp",img)
        break