export task=panoptic
BASE_CONFIG=/home/thomas/Work/Research/uil/OneFormer

python /home/thomas/Work/Research/uil/OneFormer/demo/demo.py --config-file $BASE_CONFIG/configs/cityscapes/convnext/oneformer_convnext_large_bs16_90k.yaml \
  --input /media/thomas/research_storage/streetview/jpegs_manhattan_touchdown_2021/*.jpg \
  --output /home/thomas/Work/Research/uil/streetview_data/google_images/nyc \
  --task $task \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS /home/thomas/Work/Research/uil/OneFormer/250_16_convnext_l_oneformer_cityscapes_90k.pth
