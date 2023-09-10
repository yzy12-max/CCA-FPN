# train
cd /data1/yzycode/mmdetection
#CUDA_VISIBLE_DEVICES=2 bash tools/dist_train.sh configs/pascal_voc/fcos_r50_augfpn_1x_coco.py 1
#CUDA_VISIBLE_DEVICES=2 bash tools/dist_train.sh configs/pascal_voc/fcos_r50_augfpn_1x_coco.py 1
CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh configs/pascal_voc/atss_r50_fpn_1x_coco.py 2
#bash tools/dist_train.sh configs/pascal_voc/atss_r50_fpn_1x_coco.py 2
#bash tools/dist_train.sh configs/pascal_voc/fcos_r50_fpn_1x_coco.py 2
#bash tools/dist_train.sh configs/underwater/gfl_r50_fpn_1x_coco.py 2
#CUDA_VISIBLE_DEVICES='1,2' bash tools/dist_train.sh configs/pascal_voc/retinanet_r50_augfpn_1x_coco.py 2



# 测试
#CUDA_VISIBLE_DEVICES=0,2 bash tools/dist_test.sh configs/pascal_voc/atss_r50_fpn_1x_coco.py work_dirs/underwater/atss_r50_fpn_1x_coco/atssaspp_wt20_12.pth 2 --eval bbox
#CUDA_VISIBLE_DEVICES=0,2 bash tools/dist_test.sh configs/pascal_voc/fcos_r50_fpn_1x_coco.py work_dirs/underwater/fcos_r50_fpn_1x_coco/fcos3_wt20_12.pth 2 --eval bbox
#CUDA_VISIBLE_DEVICES=0,2 bash tools/dist_test.sh configs/pascal_voc/retinanet_r50_augfpn_1x_coco.py work_dirs/underwater/retinanet_r50_augfpn_1x_coco/augfpn_wt20_12.pth 2 --eval bbox

# 检测PARAMETER
#CUDA_VISIBLE_DEVICES=2 python /data1/yzycode/mmdetection/tools/analysis_tools/get_flops.py /data1/yzycode/mmdetection/configs/pascal_voc/atss_r50_fpn_1x_coco.py

# 检测速度 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500
#CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py configs/pascal_voc/atss_r50_fpn_1x_coco.py \
#  work_dirs/underwater/atss_r50_fpn_1x_coco/atsssifpn_wt20_12.pth --launcher pytorch

#CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 /data1/yzycode/mmdetection/tools/analysis_tools/benchmark.py /data1/yzycode/mmdetection/configs/pascal_voc/atss_r50_fpn_1x_coco.py /data1/yzycode/mmdetection/work_dirs/atss_r50_fpn_1x_coco/epoch_1220220724_215255.pth --launch pytorch


# 可视化
#CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/pascal_voc/atss_r50_augfpn_1x_coco.py work_dirs/underwater/atss_r50_augfpn_1x_coco/atssaug_wt20_12.pth \
# --show-dir work_dirs/underwater/atss_r50_fpn_1x_coco/aug

# 浏览数据集,CUDA_VISIBLE_DEVICES=0,1
#python tools/misc/browse_dataset.py   configs/pascal_voc/atss_r50_fpn_1x_coco.py  --output-dir work_dirs/underwater/atss_r50_fpn_1x_coco/underwatertrainlabel --not-show
