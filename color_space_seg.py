#! /usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import image_show_utils as isu
import os
from matplotlib import pyplot as plt


def hsv_hist_extraction(hsv_hist, image_path, file_name):
	image_data = cv2.imread(image_path, -1)
	img_height, img_width, _ = image_data.shape
	y_cr_cb = cv2.cvtColor(image_data, cv2.COLOR_BGR2Lab)  # 转换至YCrCb空y间
	(y, cr, cb) = cv2.split(y_cr_cb)  # 拆分出Y,Cr,Cb值
	cr1 = cv2.GaussianBlur(cr, (3, 3), 0)

	# 这里是亮度，因此0-255 都不会影响
	_, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Ostu处理
	# isu.show_image_with_fix_windows(skin, False)

	# 反色： 黑变白，白变黑
	mask = 255 - skin
	# isu.show_image_with_fix_windows(mask, False)

	koutuRet = cv2.bitwise_and(image_data, image_data, mask=skin)
	# isu.show_image_with_fix_windows(koutuRet, True)

	cv2.imwrite('./images/final/' + file_name, koutuRet)

	# 这里要特别强调一个参数： cv2.RETR_EXTERNAL，意思是找一个外接轮廓，对于多目标而言，这个参数是非常好的。
	# 如果使用cv2.RETR_TREE，可能就会存在轮廓嵌套轮廓，而导致对于有空洞的目标不容易提取
	# contours, hierarchy = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	pass


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
	# isu.show_image_with_fix_windows(mask, False)

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
	# plt.plot(roi_hist, label='B', color='green')
	# plt.show()

	# ret 对部分区间的值reset
	for i in range(len(roi_hist)):
		if i >= 1 and i <= 20:
			continue
		else:
			roi_hist[i] = np.array(0.000000)

	return roi_hist


pass

files = os.listdir('./images/crop')

raw_image = cv2.imread('./images/crop/' + files[0], -1)
hsv_hist = cacl_hsv_hist(raw_image)

for file in files:
	full_path = './images/crop/' + file
	hsv_hist_extraction(hsv_hist, full_path, file)

# reference
# https://blog.csdn.net/wsp_1138886114/article/details/80660014
# https://juejin.cn/post/6844903631968272391
# https://blog.csdn.net/xiao__run/article/details/77135375
# https://zhuanlan.zhihu.com/p/74202427
