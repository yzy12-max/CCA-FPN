import glob, tqdm, shutil, os
import random

random.seed(11)  # 每次拿出的随机数就会相同，避免要重新划分数据集后，导致训练的数据集不同，影响结果，也方便别人根据数据集复现你的结果
# 划分数据集
if __name__ == '__main__':
    data_root = '/data/wjh/datasets/forUser_A/voice-train/image'
    image_paths = glob.glob(data_root + '/*.bmp')  # 固定随机的图像绝对路径名，不是顺序的
    # 1.U004229，U000630
    # 划分数据集， 10%作为验证集
    train_size = int(len(image_paths) * 0.85)  # 0.85为训练集
    train_paths = image_paths[:train_size]
    val_paths = image_paths[train_size:]

    os.makedirs('/data/wjh/datasets/forUser_A/voice-train/train-image/', exist_ok=True)
    os.makedirs('/data/wjh/datasets/forUser_A/voice-train/train-box/', exist_ok=True)
    os.makedirs('/data/wjh/datasets/forUser_A/voice-train/val-image/', exist_ok=True)
    os.makedirs('/data/wjh/datasets/forUser_A/voice-train/val-box/', exist_ok=True)

    for path in tqdm.tqdm(train_paths):
        base_name = os.path.basename(path)
        dst_name = os.path.join('/data/wjh/datasets/forUser_A/voice-train/train-image', base_name)
        xml_name = base_name.split('.')[0] + '.xml'
        xml_src_path = os.path.join('/data/wjh/datasets/forUser_A/voice-train/box', xml_name)  # 原来整个BOX文件夹
        xml_dst_path = os.path.join('/data/wjh/datasets/forUser_A/voice-train/train-box', xml_name)  # 训练的BOX文件夹
        shutil.copy(path, dst_name)  # 复制过去
        shutil.copy(xml_src_path, xml_dst_path)

    for path in tqdm.tqdm(val_paths):
        base_name = os.path.basename(path)
        dst_name = os.path.join('/data/wjh/datasets/forUser_A/voice-train/val-image', base_name)
        xml_name = base_name.split('.')[0] + '.xml'
        xml_src_path = os.path.join('/data/wjh/datasets/forUser_A/voice-train/box', xml_name)
        xml_dst_path = os.path.join('/data/wjh/datasets/forUser_A/voice-train/val-box', xml_name)
        shutil.copy(path, dst_name)
        shutil.copy(xml_src_path, xml_dst_path)