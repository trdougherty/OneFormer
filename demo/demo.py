# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import argparse
import multiprocessing as mp
import os
import torch
import random
import logging
# fmt: off
import sys
from pathlib import Path

sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

from panoptic_manipulation import pull_instanceinfo

import time
import cv2
import numpy as np
import tqdm
import json

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from predictor import VisualizationDemo

# constants
WINDOW_NAME = "OneFormer Demo"

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="oneformer demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="../configs/ade20k/swin/oneformer_swin_large_IN21k_384_bs16_160k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--task", help="Task type")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")

    logger = setup_logger()
    # logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, parallel=True)
    # logger.info("Config metadata: {}".format(demo.metadata))

    if args.input:
        for path in tqdm.tqdm(args.input, disable=not args.output):
            output_singlefilename = Path(path).stem
            # use PIL, to be consistent with evaluation
                
            img = read_image(path, format="BGR")
            start_time = time.time()

            opath = os.path.join(args.output, 'panoptic_inference')
            out_data_filename = opath + '_data'
            counts_out = os.path.join(out_data_filename, output_singlefilename + '_count.json')
            if not os.path.exists(counts_out):
                predictions, visualized_output, img_info = demo.run_on_image(img, args.task)
                # logger.info("Predictions: {}".format(predictions))
                # logger.info("Prediction PANO: {}".format(predictions['panoptic_seg']))
                # logger.info("New Metadata: {}".format(demo.metadata))

                counts, counts_area = pull_instanceinfo(
                    predictions['panoptic_seg'][1],
                    demo.metadata.stuff_classes,
                    img_info
                )

                # logger.info("Prediction Information: {}".format(counts))
                # logger.info("Prediction Areas {}".format(counts_area))
                # logger.info(
                #     "{}: {} in {:.2f}s".format(
                #         path,
                #         "detected {} instances".format(len(predictions["instances"]))
                #         if "instances" in predictions
                #         else "finished",
                #         time.time() - start_time,
                #     )
                # )
                if args.output:
                    # logger.info("Args: {}".format(args.input))
                    # logger.info("Visualized Output: {}".format(visualized_output))

                    if len(args.input) == 1:
                        for k in visualized_output.keys():
                            os.makedirs(k, exist_ok=True)
                            out_filename = os.path.join(k, args.output)
                            
                            out_data_filename = os.path.join(k, args.output + '_data')

                            counts_out = os.path.join(out_data_filename, output_singlefilename + '_count.json')
                            with open(counts_out, "w") as outfile:
                                json.dump(counts, outfile)

                            areas_out = os.path.join(out_data_filename, output_singlefilename + '_area.json')
                            with open(areas_out, "w") as outfile:
                                json.dump(counts_area, outfile)

                            visualized_output[k].save(out_filename)    
                    else:
                        for k in visualized_output.keys():
                            opath = os.path.join(args.output, k)    
                            os.makedirs(opath, exist_ok=True)
                            out_filename = os.path.join(opath, os.path.basename(path))
                            visualized_output[k].save(out_filename)    
                            
                            out_data_filename = opath + '_data'
                            os.makedirs(out_data_filename, exist_ok=True)

                            counts_out = os.path.join(out_data_filename, output_singlefilename + '_count.json')

                            with open(counts_out, "w") as outfile:
                                json.dump(counts, outfile)

                            areas_out = os.path.join(out_data_filename, output_singlefilename + '_area.json')
                            with open(areas_out, "w") as outfile:
                                json.dump(counts_area, outfile)
                else:
                    raise ValueError("Please specify an output path!")
            else:
                pass
    else:
        raise ValueError("No Input Given")
