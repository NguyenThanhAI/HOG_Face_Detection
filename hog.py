from typing import Tuple
import numpy as np
from scipy.signal import convolve2d
from scipy import io
import cv2
from skimage import feature


def get_differential_filter() -> Tuple[np.ndarray, np.ndarray]:
    
    filter_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    filter_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    return filter_x, filter_y


def filter_image(im: np.ndarray, filter: np.ndarray) -> np.ndarray:
    
    im_filtered: np.ndarray = convolve2d(in1=im, in2=filter, mode="same")
    
    return im_filtered.astype(np.float32)


def get_gradient(im_dx: np.ndarray, im_dy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    grad_mag, grad_angle = cv2.cartToPolar(x=im_dx, y=im_dy)
    
    grad_angle = (grad_angle / (2 * np.pi)) * 360
    
    return grad_mag, grad_angle


#def build_histogram(grad_mag, grad_angle, cell_size):
    


positive_path = r"C:\Users\Thanh_Tuyet\Downloads\practical-face-detection\practical-face-detection\images\possamples.mat"

negative_path = r"C:\Users\Thanh_Tuyet\Downloads\practical-face-detection\practical-face-detection\images\negsamples.mat"

pos_mat = io.loadmat(file_name=positive_path)["possamples"]
neg_mat = io.loadmat(file_name=negative_path)["negsamples"]

pos_mat = np.transpose(pos_mat, axes=(2, 0, 1))
neg_mat = np.transpose(neg_mat, axes=(2, 0, 1))

image = pos_mat[0]
filter_x, filter_y = get_differential_filter()
dx = filter_image(im=image, filter=filter_x)
dy = filter_image(im=image, filter=filter_y)
#dx = cv2.Sobel(image, cv2.CV_32F, dx=0, dy=1, ksize=3)
#dy = cv2.Sobel(image, cv2.CV_32F, dx=1, dy=0, ksize=3)
print("image:", image.shape)
print("dx:", dx.dtype, dy.dtype)
grad_mag, grad_angle = get_gradient(im_dx=dx, im_dy=dy)

print("grad mag:", grad_mag)
print("grad angle:", grad_angle)
cv2.imshow("Anh", image)
cv2.waitKey(0)

hog_feature = feature.hog(image=image, orientations=9, cells_per_block=(2, 2), block_norm="L2")
print("scikit image:", hog_feature)


cell_size = (8, 8)  # h x w in pixels
block_size = (2, 2)  # h x w in cells
nbins = 9  # number of orientation bins

# 2. Tính toán các tham số truyền vào HOGDescriptor
# winSize: Kích thước của bức ảnh được crop để chia hết cho cell size.
winSize = (image.shape[1] // cell_size[1] * cell_size[1], image.shape[0] // cell_size[0] * cell_size[0])
# blockSize: Kích thước của 1 block
blockSize = (block_size[1] * cell_size[1], block_size[0] * cell_size[0])
# blockStride: Số bước di chuyển của block khi thực hiện chuẩn hóa histogram bước 3
blockStride = (cell_size[1], cell_size[0])
print('Kích thước bức ảnh crop theo winSize (pixel): ', winSize)
print('Kích thước của 1 block (pixel): ', blockSize)
print('Kích thước của block stride (pixel): ', blockStride)

# 3. Compute HOG descriptor
hog = cv2.HOGDescriptor(_winSize=winSize,
                        _blockSize=blockSize,
                        _blockStride=blockStride,
                        _cellSize=cell_size,
                        _nbins=nbins)

# Kích thước của lưới ô vuông.
n_cells = (image.shape[0] // cell_size[0], image.shape[1] // cell_size[1])
print('Kích thước lưới ô vuông (ô vuông): ', n_cells)

# Reshape hog feature
hog_feats = hog.compute(image)\
               #.reshape(n_cells[1] - block_size[1] + 1,
               #         n_cells[0] - block_size[0] + 1,
               #         block_size[0], block_size[1], nbins)
               
               
print("OpenCV: ", hog_feats)