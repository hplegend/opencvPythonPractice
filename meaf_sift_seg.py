#! /usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import image_show_utils as isu
import os
from matplotlib import pyplot as plt


def cacl_hsv_hist(image_data):
	img_height, img_width, _ = image_data.shape
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
	mask = cv2.inRange(hsv_roi, np.array((0., 30., 32.)), np.array((180., 255., 255.)))
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
	hsv_roi_test = cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)

	# 反向投影是基于概率的
	dst = cv2.calcBackProject([hsv_roi_test], [0], roi_hist, [0, 180], 1)
	blur_ret = cv2.blur(dst, (3, 3))
	median_ret = cv2.medianBlur(blur_ret, 3)

	# isu.show_image_with_fix_windows(dst, False)
	isu.show_image_with_fix_windows(median_ret, False)

	# 开始用反向投影作为模板抠图
	empty_new_image = np.zeros((img_height, img_width, 3), dtype='uint8') + 255
	# mask_inv = cv2.bitwise_not(maskImage)
	extract_mask_image = cv2.bitwise_and(image_data, empty_new_image, mask=blur_ret)

	# 剔除小轮廓
	gray = cv2.cvtColor(extract_mask_image, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)

	isu.show_image_with_fix_windows(thresh, False)

	_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	reserve_contours = []

	for ele_contour in contours:
		area = cv2.contourArea(ele_contour)
		if area > 0 and area < 50:
			continue
		if area > img_width * img_height * 0.8:
			continue

		reserve_contours.append(ele_contour)

	print(len(reserve_contours))

	reserve_contour_image = np.zeros((img_height, img_width, 3), dtype='uint8') + 0
	cv2.drawContours(reserve_contour_image, reserve_contours, -1, (255, 255, 255), thickness=-1)
	isu.show_image_with_fix_windows(reserve_contour_image, True)

	isu.show_image_with_fix_windows(extract_mask_image, True)


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
