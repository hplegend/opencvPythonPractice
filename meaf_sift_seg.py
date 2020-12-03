#! /usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import image_show_utils as isu
import os
from matplotlib import pyplot as plt


def cacl_hsv_hist(image_data):
	hsv_roi = cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)
	# isu.show_image_with_fix_windows(hsv_roi)

	# 单独拎出来色相图
	img_h = hsv_roi[..., 0]
	img_s = hsv_roi[..., 1]
	img_v = hsv_roi[..., 2]

	img_b = image_data[..., 0]
	img_g = image_data[..., 1]
	img_r = image_data[..., 2]

	# isu.show_image_with_fix_windows(img_h, False)
	# isu.show_image_with_fix_windows(img_s)
	# isu.show_image_with_fix_windows(img_v)

	# 过滤
	# 相当于阈值分割那么一回事
	# mask 已经是灰度图了
	mask = cv2.inRange(hsv_roi, np.array((0., 40., 32.)), np.array((180., 255., 255.)))
	isu.show_image_with_fix_windows(mask, False)

	roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
	plt.plot(roi_hist, label='B', color='red')
	plt.show()

	"""
	images:   图片列表
	channels: 需要计算直方图的通道。[0]表示计算通道0的直方图，[0,1,2]表示计算通道0,1,2所表示颜色的直方图
	mask:     蒙版，只计算值>0的位置上像素的颜色直方图，取None表示无蒙版
	histSize: 每个维度上直方图的大小，[8]表示把通道0的颜色取值等分为8份后计算直方图
	ranges:   每个维度的取值范围，[lower0, upper0, lower1, upper1, ...]，lower可以取到，upper无法取到
	hist:     保存结果的ndarray对象
	accumulate: 是否累积，如果设置了这个值，hist不会被清零，直方图结果直接累积到hist中
	"""

	roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
	# 归一化
	cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
	plt.plot(roi_hist, label='B', color='green')
	plt.show()

	# ret 对部分区间的值reset
	for i in range(len(roi_hist)):
		if i >= 1 and i <= 20:
			continue
		else:
			roi_hist[i] = np.array(0.000000)

	# 选择大概率的，然后进行反向投影
	# dst 应该是概率图
	new_raw_image = cv2.imread('./images/crop/IMG_20181003_140313.jpg', -1)
	hsv_roi_test = cv2.cvtColor(new_raw_image, cv2.COLOR_BGR2HSV)

	# 反向投影是基于概率的
	dst = cv2.calcBackProject([hsv_roi_test], [0], roi_hist, [0, 180], 1)

	isu.show_image_with_fix_windows(dst, False)

	plt.plot(roi_hist, label='B', color='blue')
	plt.show()
	pass


files = os.listdir('./images/crop')

for file in files:
	raw_image = cv2.imread('./images/crop/' + file, -1)
	cacl_hsv_hist(raw_image)

# reference
# https://blog.csdn.net/wsp_1138886114/article/details/80660014
# https://juejin.cn/post/6844903631968272391
# https://blog.csdn.net/xiao__run/article/details/77135375
# https://zhuanlan.zhihu.com/p/74202427
