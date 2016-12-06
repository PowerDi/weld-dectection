# -*- coding: utf8 -*-
__author__ = 'PowerDi'
__address__ = '578661971@qq.com'

import cv2
import numpy as np
#手动更换下面图片的路径+文件名
'''''
经常用到的颜色空间转换是: BGR<->Gray 和 BGR<->HSV
cv2.cvtColor(input_image , flag),flag是转换类型：cv2.COLOR_BGR2GRAY,cv2.COLOR_BGR2HSV

HSV(Hue , Saturation , Value):色调，饱和度，明度
色度H:用角度度量，取值范围为0~360，红色开始按逆时针方向计算，红色为0度，绿色为120度，蓝色为240度
饱和度S:接近光谱色的程度，颜色可以看成是光谱色与白色混合结果，光谱色占的比例愈大，颜色接近光谱色的程度
        越高，颜色饱和度就越高。光谱色中白色成分为0，饱和度达到最高，取值范围0%~100%，值越大，颜色越饱和
明度V:表示颜色明亮的程度，对于光源色，明度值与发光体的光亮度有关；对于物体色，与物体的透射比有关，取值
      范围为0%(黑)~100%(白)
RGB面向硬件，HSV面向用户

在Opencv中
H色度取值范围是[0,179]
S饱和度的取值范围是[0,255]
V明度的取值范围是[0,255]

'''
def BIAOZHU(fn):
    '''
    Label the contours of a weld
    :param fn: 'afterlabel.bmp'
    :return: None
    '''
    image_init = cv2.imread('test.bmp')# initial picture
    image_biaozhu = image_init.copy()
    image_blue = cv2.imread(fn)
    frame = image_blue.copy()
    # 转到HSV空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    '''''
    cv2.inRange(src , lowerb , upperb[,dst])
    作用：更改函数对某个单通道中的元素检查其值是否在范围中
    src:输入数组，lowerb:包含低边界的数组，upperb:包含高边界的数组，dst:输出数组
    如果src(I)符合范围，则dst(I)被设置为255，也就是说dst返回的是非黑即白的图像，而且符合要求
    的部分是白色的
    '''
    #构建物体掩膜（黑白部分），注意这里要使用hsv
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    #对原图像和掩膜进行位运算
    res = cv2.bitwise_and(frame, frame, mask=mask)
    '''
    取边缘
    '''
    #对区域进行腐蚀操作
    erode_mask = cv2.erode(mask, None, iterations=1)
    #找出黑白边缘,第二个参数表示树结构;第三个参数表示只选出能连成边缘的几个点,而不是整个边缘
    contours, hierarchy = cv2.findContours(erode_mask.copy(),
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_biaozhu, contours, -1, (0, 0, 255), 3)
    cv2.imshow("image_biaozhu", image_biaozhu)
    cv2.imwrite("biaozhu.bmp", image_biaozhu) #换成你想存放的标注图的路径
    cv2.waitKey()
    cv2.destroyAllWindows()

fn='afterlabel.bmp'
BIAOZHU(fn)