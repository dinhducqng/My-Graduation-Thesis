import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from numpy import mean
import sys
from  PIL  import Image

# Đọc ảnh
img = cv.imread('E:/tomatoes/ca chua do/ca chua 7 ( 1 ).bmp')

# Resize 250x250 pixel

img_rs =cv.resize(img,dsize=(250,250))
original = img_rs.copy()
result = original

# Tách nền xanh dương, ra vật
#  Convert sang HSV để tách, vì nền là màu xanh đối lập hsv với đỏ

image = cv.cvtColor(original, cv.COLOR_BGR2HSV)
lower = np.array([80,0,0])
upper = np.array([130,255,255])
mask = cv.inRange(image, lower, upper)
mask = cv.bitwise_not(mask)

# tạo nền mới
bg_image = np.zeros([img_rs.shape[0],img_rs.shape[1],img_rs.shape[2]],dtype=np.uint8)
condition = np.stack(
    (mask,) * 3, axis=-1) > 0.5

# Hoàn tất tạo nền mới

output_image = np.where(condition, img_rs, bg_image)
cv.imshow('Hinh da tach',output_image)

# Convert sang hệ màu HSV để trích đặc trưng
#result = cv.cvtColor(result, cv.COLOR_BGR2HSV)
img_hsv = cv.cvtColor(output_image, cv.COLOR_BGR2HSV)

# Đặt tham số để vẽ Histogram
# HSV của tổng quả
H, S, V = img_hsv[:,:,0],img_hsv[:,:,1],img_hsv[:,:,2]

# Tính trung bình H của cả quả

dataH_Tong = [img_hsv[:,:,0]]
avgH_Tong = mean(dataH_Tong)
print("avgH_Tong",round(avgH_Tong))

# Tính trung bình S của cả quả

dataS_Tong = [img_hsv[:,:,1]]
avgS_Tong= mean(dataS_Tong)
print("avgS_Tong",round(avgS_Tong))

# Tính trung bình V cả quả

dataV_Tong = [img_hsv[:,:,2]]
avgV_Tong= mean(dataV_Tong)
print("avgV_Tong",round(avgV_Tong))

# Ảnh đã tách

result[mask ==  0] =  [0]

# Cắt làm 3 phần top, mid và bot

cut_num = int(round(img_hsv.shape[0]/3))

# Cắt ra 3 phần và xuất Histogram

top = img_hsv[0:cut_num, :, :]
mid = img_hsv[cut_num:cut_num*2,:,:]
bottom = img_hsv[cut_num*2:,:,:]

# H,S,V của mặt top

H_top, S_top, V_top = top[:,:,0],top[:,:,1],top[:,:,2]

# Tính trung bình H top

dataH_Top = [top[:,:,0]]
avgH_Top= mean(dataH_Top)
print("avgH_Top",round(avgH_Top))

# Tính trung bình S top

dataS_Top = [top[:,:,1]]
avgS_Top= mean(dataS_Top)
print("avgS_Top",round(avgS_Top))

# Tính trung bình V top

dataV_Top = [top[:,:,2]]
avgV_Top= mean(dataV_Top)
print("avgV_Top",round(avgV_Top))

# H,S,V của mặt mid

H_mid, S_mid, V_mid = mid[:,:,0],mid[:,:,1],mid[:,:,2]

# Tính trung bình H mid

dataH_mid = [mid[:,:,0]]
avgH_mid= mean(dataH_mid)
print("avgH_mid",round(avgH_mid))

# Tính trung bình S mid

dataS_mid = [mid[:,:,1]]
avgS_mid= mean(dataS_mid)
print("avgS_mid",round(avgS_mid))

# Tính trung bình V mid

dataV_mid = [mid[:,:,2]]
avgV_mid= mean(dataV_mid)
print("avgV_mid",round(avgV_mid))

# H,S,V của mặt bottom

H_bottom, S_bottom, V_bottom = bottom[:,:,0],bottom[:,:,1],bottom[:,:,2]

# Tính trung bình H bottom

dataH_bottom = [bottom[:,:,0]]
avgH_bottom= mean(dataH_bottom)
print("avgH_bottom",round(avgH_bottom))

# Tính trung bình S bottom

dataS_bottom = [bottom[:,:,1]]
avgS_bottom = mean(dataS_bottom)
print("avgS_bottom",round(avgS_bottom))

# Tính trung bình V bottom

dataV_bottom = [bottom[:,:,2]]
avgV_bottom= mean(dataV_bottom)
print("avgV_bottom",round(avgV_bottom))

# Xuất biểu đồ Histogram
plt.figure(figsize=(10,8))
plt.subplot(311)                             #Biểu đồ cột Hue
plt.subplots_adjust(hspace=.5)
plt.title("Hue")
plt.hist(np.ndarray.flatten(H), bins=180,range=(1,255))
plt.subplot(312)                             #Biểu đồ cột Saturation
plt.title("Saturation")
plt.hist(np.ndarray.flatten(S), bins=128,range=(1,255))
plt.subplot(313)                             #Biểu đồ cột Value
plt.title("Luminosity Value")
plt.hist(np.ndarray.flatten(V), bins=128,range=(1,255))
plt.show()

# Show ảnh
cv.imshow('anh goc',img_rs)
cv.imshow('anh xu ly',result)
cv.imshow('anh hsv',img_hsv)
cv.imshow('top',top)
cv.imshow('mid',mid)
cv.imshow('bottom',bottom)
cv.waitKey()
cv.destroyAllWindows()
