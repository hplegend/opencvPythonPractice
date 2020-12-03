#! /usr/bin/env python
# coding: utf-8

import cv2


def show_image_with_fix_windows(image_data, has_more_channel):
	# scale ratio
	scale_ratio = 0.2
	# 等比
	if has_more_channel:
		h, w, _ = image_data.shape
	else:
		h, w = image_data.shape

	window_h = int(h * scale_ratio)
	window_w = int(w * scale_ratio)

	cv2.namedWindow('fix_windows', 0)
	cv2.resizeWindow("fix_windows", window_w, window_h)
	cv2.moveWindow("fix_windows", 20, 30)
	cv2.imshow('fix_windows', image_data)
	cv2.waitKey()
	cv2.destroyWindow('fix_windows')
