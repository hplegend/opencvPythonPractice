#! /usr/bin/env python
# coding: utf-8
# -*- coding: utf-8 -*-

# 交互式提取target
# mean shift跟踪
# 阈值分割
# 写入结果

import cv2
import os
import numpy as np
import opencv_iterative_utils as oiu


# return ltx,lty,lrx,lry
def get_first_trace_rect(file_name):
	return oiu.get_rect_post_iteractive(file_name)
	pass


def simple_crop(initalSx, initalSy, initalEx, initalEy):
	filesName = os.listdir('./images/raw/')
	w = initalEx - initalSx
	h = initalEy - initalSy
	for file in filesName:
		frame = cv2.imread('./images/raw/' + file, 1)
		croped = frame[initalSy:initalSy + h, initalSx: initalSx + w]
		# cv2.imshow('croped', croped)
		# cv2.waitKey()
		cv2.imwrite('./images/crop/' + file, croped)

	pass


def mean_shift_object_track(frame, initalSx, initalSy, initalEx, initalEy):
	# 设置初试窗口位置和大小
	r, c = initalSx, initalSy
	h, w = initalEy - initalSy, initalEx - initalSx

	print(r, c, h, w)

	track_window = (r, c, h, w)

	# 设置追踪的区域
	roi = frame[r:r + w, c:c + h]
	# roi区域的hsv图像
	hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
	# 取值hsv值在(0,60,32)到(180,255,255)之间的部分
	mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
	# 计算直方图,参数为 图片(可多)，通道数，蒙板区域，直方图长度，范围
	roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
	# 归一化
	cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

	# 设置终止条件，迭代10次或者至少移动1次
	term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 20)
	filesName = os.listdir('./images/raw/')
	for file in filesName:
		frame = cv2.imread('./images/raw/' + file, 1)
		# 计算每一帧的hsv图像
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		# 计算反向投影
		dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

		# 调用meanShift算法在dst中寻找目标窗口，找到后返回目标窗口
		ret, track_window = cv2.meanShift(dst, track_window, term_crit)
		# Draw it on image
		x, y, w, h = track_window
		# img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
		# cv2.imshow('img2',img2)
		# cv2.waitKey()
		croped = frame[x: x + w, y:y + h]
		# cv2.imshow('croped', croped)
		# cv2.waitKey()
		cv2.imwrite('./images/crop/' + file, croped)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		cv2.destroyAllWindows()


def target_extract():
	# 模板抠图
	filesName = os.listdir('./images/crop/')
	for file in filesName:
		print('./images/crop/' + file)
		frame = cv2.imread('./images/crop/' + file, -1)
		h, w, _ = frame.shape
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

		cv2.namedWindow('thresh', 0)
		cv2.resizeWindow("thresh", 756, 756)
		cv2.moveWindow("thresh", 20, 30)
		cv2.imshow('thresh', thresh)
		cv2.waitKey()

		emptyNewImage = np.zeros((h, w, 3), dtype='uint8') + 255
		# mask_inv = cv2.bitwise_not(maskImage)
		rawMaskInamge = cv2.bitwise_and(frame, emptyNewImage, mask=thresh)

		cv2.namedWindow('finalRet', 0)
		cv2.resizeWindow("finalRet", 756, 756)
		cv2.moveWindow("finalRet", 20, 30)
		cv2.imshow('finalRet', rawMaskInamge)
		cv2.waitKey()
		cv2.imwrite('./images/final/' + file, rawMaskInamge)

	# 阈值图像作为mask抠图

	pass


def mean_shift_trace_and_crop(file_path, first_file_name):
	ltx, lty, lrx, lry = get_first_trace_rect(first_file_name)
	first_image = cv2.imread(first_file_name, -1)
	simple_crop(ltx, lty, lrx, lry)
	target_extract()

	pass


mean_shift_trace_and_crop('./images/', './images/raw/IMG_20181003_135915.jpg')
