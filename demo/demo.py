# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import csv
import configparser
import pandas as pd

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args, csv_cfg):
    # load config from file and command-line arguments
    cfg = get_cfg()
    args.config_file = csv_cfg.get("DEFAULT","pretrain_config")
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default= "configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument("--mask-path",help="root file to save mask results",default='/workspace/regression_data/masks')
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--code",
        help="the name of the report",
        default='result_demo_0225'
    )

    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    csv_cfg = configparser.ConfigParser()  
    csv_cfg.read('../config_csv.yaml')

    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    csv_output_folder = csv_cfg.get("DEFAULT","OUTPUT_FOLDER")
    csv_true = False
    if csv_cfg.get("DEFAULT", "OUTPUT"):
        f = open(csv_output_folder,'w', newline='')
        wr = csv.writer(f)
        csv_true = True

    mask_true = False
    if csv_cfg.get("DEFAULT","MASK"):
        mask_true = True
    path_true = False
    if csv_cfg.get("DEFAULT", "IMAGE_PATH"):
        path_true = True
    cfg = setup_cfg(args, csv_cfg)

    demo = VisualizationDemo(cfg)
    mask_path = args.mask_path
    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output, mask, pred_classes = demo.run_on_image(img, path,mask_path)

            img_path = path
            instance_num = len(predictions["instances"])
            row = []
            if csv_true:
                row.append(path)
                row.append(instance_num)
            if mask_true:
                row.append(pred_classes)
                
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(predictions["instances"]), time.time() - start_time
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    os.mkdir(args.output)
                    # assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
                print(f"visualized output saved to {out_filename}")
                if path_true:
                    row.append(out_filename)
            else:
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
            wr.writerow(row)
        f.close()
        logger.info(
                "report saved to {}".format(
                    csv_output_folder
                )
            )
            

    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
