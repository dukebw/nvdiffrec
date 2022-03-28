"""Extract foreground masks as preprocessing for nvdiffrec training data

Based on detectron2/demo/demo.py
"""
import multiprocessing as mp

import click
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger


@click.command()
@click.option(
    "--confidence-threshold",
    type=float,
    default=0.5,
    help="Minimum score for instance predictions to be shown",
)
@click.option(
    "--config-path",
    type=str,
    required=True,
    help="Path to config file",
)
@click.option(
    "--input-img-dir",
    type=str,
    required=True,
    help="Path to directory containing images to process",
)
@click.option(
    "--opts",
    type=str,
    default="",
    help="Modify config options using the command-line 'KEY VALUE' pairs",
)
def extract_masks(
    confidence_threshold: float, config_path: str, input_img_dir: str, opts: str
):
    """Extract foreground masks"""
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(opts.split())
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()

    predictor = DefaultPredictor(cfg)


if __name__ == "__main__":
    extract_masks()
