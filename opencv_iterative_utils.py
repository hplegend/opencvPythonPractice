#! /usr/bin/env python
# coding: utf-8

import cv2
import copy

SelectFirstX = 0
SelectFirstY = 0
SelectSecondX = 0
SelectSecondY = 0
MoveX = 0
MoveY = 0
DrawFinish = False
BeginDraw = False


# OpenCV鼠标回调处理函数
def mouse_call_back(event, x, y, flags, param):
	global SelectFirstX, SelectFirstY
	global SelectSecondX, SelectSecondY
	global DrawFinish, BeginDraw
	global MoveX, MoveY
	if event == cv2.EVENT_LBUTTONUP:
		BeginDraw = True
		SelectFirstX = x
		SelectFirstY = y
		MoveX = x
		MoveY = y

	if event == cv2.EVENT_RBUTTONUP:
		DrawFinish = True
		SelectSecondX = x
		SelectSecondY = y

	if event == cv2.EVENT_MOUSEMOVE:
		MoveX = x
		MoveY = y


# 拾取两个点，分别为左右位置
def get_rect_post_iteractive(fileName):
	global DrawFinish
	raw_image = cv2.imread(fileName, -1)
	cv2.namedWindow('iteractive', 0)
	cv2.resizeWindow("iteractive", 756, 756)
	cv2.moveWindow("iteractive", 20, 30)
	cv2.setMouseCallback('iteractive', mouse_call_back)
	cv2.imshow('iteractive', raw_image)
	cv2.waitKey()

	DrawFinish = False
	# 这里刻意要暂停程序的执行，等待鼠标操作完毕
	while DrawFinish == False:
		cv2.waitKey(10)
		copy_image = copy.copy(raw_image)
		image = cv2.rectangle(copy_image, (SelectFirstX, SelectFirstY),
		                      (MoveX, MoveY), (255, 0, 0), 4)
		cv2.imshow('iteractive', image)

	copy_image = copy.copy(raw_image)
	image = cv2.rectangle(copy_image, (SelectFirstX, SelectFirstY),
	                      (MoveX, MoveY), (255, 0, 0), 4)

	cv2.imshow('iteractive', image)
	cv2.waitKey()
	cv2.destroyAllWindows()
	print(MoveX, MoveY)
	return SelectFirstX, SelectFirstY, SelectSecondX, SelectSecondY
