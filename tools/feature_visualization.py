#自己增加的特征图可视化
#https://blog.csdn.net/H_zzz_W/article/details/115316820
import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    # heatmap = feature_map[:,0,:,:]*0    #
    heatmap = feature_map[:1, 0, :, :] * 0 #取一张图片
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:1,c,:,:]
    heatmap = heatmap.cpu().numpy()    #因为数据原来是在GPU上的
    heatmap = np.mean(heatmap, axis=0) #计算像素点的平均值,会下降一维度

    heatmap = np.maximum(heatmap, 0)  #返回大于0的数
    heatmap /= np.max(heatmap)      #/最大值来设置透明度0-1
    heatmaps.append(heatmap)

    return heatmaps

def featuremap_2_heatmap1(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    # heatmap = feature_map[:,0,:,:]*0    #
    heatmap = feature_map[:1, 0, :, :] * 0 #取一张图片,初始化为0
    for c in range(feature_map.shape[1]):   # 按通道
        heatmap+=feature_map[:1,c,:,:]      # 像素值相加[1,H,W]
    heatmap = heatmap.cpu().numpy()    #因为数据原来是在GPU上的
    heatmap = np.mean(heatmap, axis=0) #计算像素点的平均值,会下降一维度[H,W]

    heatmap = np.maximum(heatmap, 0)  #返回大于0的数[H,W]
    heatmap /= np.max(heatmap)      #/最大值来设置透明度0-1,[H,W]
    #heatmaps.append(heatmap)

    return heatmap

def draw_feature_map(features,save_dir = './work_dirs/feature_map',name = 'feature_'):
    i=0
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)    #增加一维
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (h, w))
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                plt.imshow(superimposed_img,cmap='gray')
                plt.show()
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) #
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                plt.imshow(superimposed_img) #,cmap='gray'
                plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)     #这里通过安键盘取消显示继续运行。
                # cv2.destroyAllWindows()
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(os.path.join(save_dir,name +str(i)+'.png'), superimposed_img)
                print(os.path.dirname(os.path.join(save_dir,name +str(i)+'.png')))
                i=i+1

def draw_feature_map1(features, img_path, save_dir = './work_dirs/feature_map/',name = None):
    '''
    :param model: 加载了参数的模型
    :param img_path: 测试图像的文件路径
    :param save_dir: 保存生成图像的文件夹
    :return:
    '''
    img = cv2.imread(img_path)      #读取文件路径
    i=0
    if isinstance(features,torch.Tensor):   # 如果是单层
        features = [features]       # 转为列表
    for featuremap in features:     # 循环遍历
        heatmap = featuremap_2_heatmap1(featuremap)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap0 = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式,0-255,heatmap0显示红色为关注区域，如果用heatmap则蓝色是关注区域
        heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = heatmap * 0.5 + img  # 这里的0.5是热力图强度因子
        plt.imshow(heatmap0)  # ,cmap='gray'
        # plt.imshow(superimposed_img)  # ,cmap='gray'
        plt.close()
        # 下面这些是对特征图进行保存，使用时取消注释
        # cv2.imshow("1",superimposed_img)
        # cv2.waitKey(0)     #这里通过安键盘取消显示继续运行。
        # cv2.destroyAllWindows()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, name + str(i) + '.png'), superimposed_img)
        cv2.imwrite(os.path.join(save_dir, name + str(i) +'hp'+ '.png'), heatmap)
        print(os.path.join(save_dir, name + str(i) + '.png'))
        i = i + 1


#在mmdet/models/detectors下面的文件中找到你所用的detector里的extract_feat()函数
#https://blog.csdn.net/H_zzz_W/article/details/115316820
def feature_map(features,save_dir='work_dirs/feature_map',name=None):
    a = torch.squeeze(features[0][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    b = torch.squeeze(features[1][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    c = torch.squeeze(features[2][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    #d = torch.squeeze(features[3][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    # for i in range(12):
    #     plt.figure(1)
    #     plt.subplot(3, 4, i + 1)
    #     plt.title('a')
    #     plt.imshow(a[:, :, i], cmap='gray')
    #     plt.figure(2)
    #     plt.subplot(3, 4, i + 1)
    #     plt.title('b')
    #     plt.imshow(b[:, :, i], cmap='gray')
    #     plt.figure(3)
    #     plt.subplot(3, 4, i + 1)
    #     plt.title('c')
    #     plt.imshow(c[:, :, i], cmap='gray')
    #     plt.figure(4)
    #     plt.subplot(3, 4, i + 1)
    #     plt.title('d')
    #     plt.imshow(d[:, :, i], cmap='gray')
    # plt.show()
    for j,x in enumerate([a,b,c]): # [a,b,c,d]
        plt.figure()
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.title(str(j))
            plt.imshow(x[:, :, i]) #, cmap='gray'
        # plt.savefig(os.path.join(save_dir,  name+str(j) + '.png'))
    plt.show()

# def feature_map(features,save_dir = 'work_dirs/feature_map',name = 'pltfeature_'):
#     a = torch.squeeze(features[0], dim=0).permute(1, 2, 0).detach().cpu().numpy()
#     b = torch.squeeze(features[1], dim=0).permute(1, 2, 0).detach().cpu().numpy()
#     c = torch.squeeze(features[2], dim=0).permute(1, 2, 0).detach().cpu().numpy()
#     d = torch.squeeze(features[3], dim=0).permute(1, 2, 0).detach().cpu().numpy()
#     for j,x in enumerate([a,b,c,d]):
#         plt.figure(j)
#         for i in range(12):
#             plt.subplot(3, 4, i + 1)
#             plt.title(str(j))
#             plt.imshow(x[:, :, i], cmap='gray')
#         plt.savefig(os.path.join(save_dir,  name+str(j) + '.png'))
#     plt.show()

def feature_map_channel(features,img_path,save_dir = 'work_dirs/feature_map',name = 'noresbnsie2ltft_'):
    a = torch.squeeze(features[0][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    b = torch.squeeze(features[1][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    c = torch.squeeze(features[2][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    d = torch.squeeze(features[3][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    img = cv2.imread(img_path)
    for j,x in enumerate([a]):
        # plt.figure()
        # print(x.shape[-1])
        for i in range(x.shape[-1]):
            heatmap = x[:, :, i]
            # heatmap = np.maximum(heatmap, 0)
            # heatmap /= np.max(heatmap)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            heatmap0 = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式,0-255,heatmap0显示红色为关注区域，如果用heatmap则蓝色是关注区域
            heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)
            superimposed_img = heatmap * 0.4 + img  # 将热力图应用于原始图像
            # plt.figure()  #
            # plt.title(str(j))
            # plt.imshow(heatmap0) #, cmap='gray'
            # # plt.savefig(os.path.join(save_dir,  name+str(j)+str(i) + '.png'))
            # plt.close()
            cv2.imwrite(os.path.join(save_dir, name + str(j)+str(i) + '.png'), superimposed_img)

def feature_single(feature):
    a = torch.squeeze(feature[0][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title("upsamelping")
        plt.imshow(a[:, :, i]) #, cmap='gray'
    plt.show()