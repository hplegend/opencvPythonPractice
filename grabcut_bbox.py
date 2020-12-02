import numpy as np
import cv2
from matplotlib import pyplot as plt

import opencv_iterative_utils as oit

ltx, lty, rbx, rby = oit.get_rect_post_iteractive('./images/one.jpg')
# 导入类库

import numpy as np

import cv2

# 画图类库，很好用，不用自己从头编写了

from matplotlib import pyplot as plt

# 导入图像（小熊）

img = cv2.imread('./images/one.jpg')

# 建立一个和img图像一样大的蒙版

mask = np.zeros(img.shape[:2], np.uint8)

# 画一个矩形框框，选出前景物体（小熊）

rect = (ltx, lty, rbx, rby)

# 建立背景模型和前景模型，大小为1*65

bgdModel = np.zeros((1, 65), np.float64)

fgdModel = np.zeros((1, 65), np.float64)

# 矩形框选法

# 把上面的数据导入GrabCut算法中，进行计算，

# 该方法参数为：1.原始图像 2.蒙版 3.矩形框框 4.背景模型 5.前景模型 6.迭代次数 7.方法选择（矩形框选法）

cv2.grabCut(img.copy(), mask, rect, bgdModel, fgdModel, 30, cv2.GC_INIT_WITH_RECT)

# 把2变为0，把3变为1

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# 将蒙版与原图做点对点乘积

img = img * mask2[:, :, np.newaxis]

# 非矩形框，蒙版法再处理

img2 = cv2.imread('./images/one.jpg')

# 导入蒙版

img3 = cv2.imread('./images/onemask001.jpg', 0)

# 把蒙版中白色地方置为1，作为确定前景。黑色地方置为0，作为确定背景

mask[img3 == 0] = 0

mask[img3 == 255] = 1

# 把上面的数据导入GrabCut算法中，进行计算，

# 该方法参数为：1.原始图像 2.蒙版 3.矩形框框（无） 4.背景模型 5.前景模型 6.迭代次数 7.方法选择（蒙版法）

cv2.grabCut(img2, mask, None, bgdModel, fgdModel, 30, cv2.GC_INIT_WITH_MASK)

# 把2变为0，把3变为1

mask3 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# 将蒙版与原图做点对点乘积

img2 = img2 * mask3[:, :, np.newaxis]

# 绘制出蒙版法处理后的图像

plt.subplot(121), plt.imshow(img2)

plt.title("grabcut-mask"), plt.xticks([]), plt.yticks([])

# 绘制出矩形框法处理后的图像

plt.subplot(122), plt.imshow(img)

plt.title("grabcut-rect"), plt.xticks([]), plt.yticks([])

# 在窗口中显示图像

plt.show()

cv2.waitKey()

cv2.destroyAllWindows()
