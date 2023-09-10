import cv2
import matplotlib.pyplot as plt


def em(imgpath):
    img = cv2.imread(imgpath)
    img1 = cv2.split(img)   # 使用 cv2.split() 分割 BGR 图像
    imgs2 = []
    for i in range(3):
        img2 = cv2.equalizeHist(img1[i])    # 将 cv2.equalizeHist() 函数应用于每个通道
        imgs2.append(img2)
    imgBGR = cv2.merge(imgs2)       # 使用 cv2.merge() 合并所有结果通道
    return imgBGR


def equalize_hist_color_hsv(imgpath):
    img = cv2.imread(imgpath)
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
    return eq_image

def cm(img):
    h,w,_= img.shape
    for i in range(h):
        for j in range(w):
            img[j][1] = abs(0.60*img[j][0]+0)
            img[j][2] = abs(0.10 * img[j][0] + 0)
            img[j][0] = abs(1.00 * img[j][0] + 0)
    return img


if __name__ == '__main__':

    imgpath = "/home/yzy/datasets/yolo_voice/images/00003.bmp"
    img = cv2.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgE = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # imgE = equalize_hist_color_hsv(imgpath)
    # # imgC = cm(imgE)
    # imgE = cv2.cvtColor(imgE,cv2.COLOR_BGR2RGB)
    plt.imshow(imgE)  #
    plt.close()
    imgE = cv2.applyColorMap(imgE, cv2.COLORMAP_JET)
    # imgC = cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)
    plt.imshow(imgE)  # ,cmap='gray'
    plt.show()
