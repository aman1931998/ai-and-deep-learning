#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import itertools
import numpy as np
import os
import shutil
import tensorflow as tf
import cv2
import tqdm

import tensorpack.utils.viz as tpviz
from tensorpack.predict import MultiTowerOfflinePredictor, OfflinePredictor, PredictConfig
from tensorpack.tfutils import SmartInit, get_tf_version_tuple
from tensorpack.tfutils.export import ModelExporter
from tensorpack.utils import fs, logger

from dataset import DatasetRegistry, register_coco, register_balloon
from config import config as cfg
from config import finalize_configs
from data import get_eval_dataflow, get_train_dataflow
from eval import DetectionResult, multithread_predict_dataflow, predict_image
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel
from viz import (
    draw_annotation, draw_final_outputs, draw_predictions,
    draw_proposal_recall, draw_final_outputs_blackwhite)


def do_visualize(model, model_path, nr_visualize=100, output_dir='output'):
    """
    Visualize some intermediate results (proposals, raw predictions) inside the pipeline.
    """
    df = get_train_dataflow()
    df.reset_state()

    pred = OfflinePredictor(PredictConfig(
        model=model,
        session_init=SmartInit(model_path),
        input_names=['image', 'gt_boxes', 'gt_labels'],
        output_names=[
            'generate_{}_proposals/boxes'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'generate_{}_proposals/scores'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'fastrcnn_all_scores',
            'output/boxes',
            'output/scores',
            'output/labels',
        ]))

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    fs.mkdir_p(output_dir)
    with tqdm.tqdm(total=nr_visualize) as pbar:
        for idx, dp in itertools.islice(enumerate(df), nr_visualize):
            img, gt_boxes, gt_labels = dp['image'], dp['gt_boxes'], dp['gt_labels']

            rpn_boxes, rpn_scores, all_scores, \
                final_boxes, final_scores, final_labels = pred(img, gt_boxes, gt_labels)

            # draw groundtruth boxes
            gt_viz = draw_annotation(img, gt_boxes, gt_labels)
            # draw best proposals for each groundtruth, to show recall
            proposal_viz, good_proposals_ind = draw_proposal_recall(img, rpn_boxes, rpn_scores, gt_boxes)
            # draw the scores for the above proposals
            score_viz = draw_predictions(img, rpn_boxes[good_proposals_ind], all_scores[good_proposals_ind])

            results = [DetectionResult(*args) for args in
                       zip(final_boxes, final_scores, final_labels,
                           [None] * len(final_labels))]
            final_viz = draw_final_outputs(img, results)

            viz = tpviz.stack_patches([
                gt_viz, proposal_viz,
                score_viz, final_viz], 2, 2)

            if os.environ.get('DISPLAY', None):
                tpviz.interactive_imshow(viz)
            cv2.imwrite("{}/{:03d}.png".format(output_dir, idx), viz)
            pbar.update()


def do_evaluate(pred_config, output_file):
    num_tower = max(cfg.TRAIN.NUM_GPUS, 1)
    graph_funcs = MultiTowerOfflinePredictor(
        pred_config, list(range(num_tower))).get_predictors()

    for dataset in cfg.DATA.VAL:
        logger.info("Evaluating {} ...".format(dataset))
        dataflows = [
            get_eval_dataflow(dataset, shard=k, num_shards=num_tower)
            for k in range(num_tower)]
        all_results = multithread_predict_dataflow(dataflows, graph_funcs)
        output = output_file + '-' + dataset
        DatasetRegistry.get(dataset).eval_inference_results(all_results, output)


def do_predict(predictor, input_file):
    try:
        img = cv2.imread(os.path.join('test_images', input_file), cv2.IMREAD_COLOR)
    
        results = predict_image(img, predictor)
        if cfg.MODE_MASK:
            final = draw_final_outputs_blackwhite(img, results)
        else:
            final = draw_final_outputs(img, results)
        viz = final.copy() #np.concatenate((img, final), axis=1) #concatenate hata dena 
        opp = cv2.imwrite(os.path.join(os.getcwd(), 'test_inferences', input_file.split('.')[0]+".png"), viz)
        if opp:
            logger.info("Inference output for {} Successful".format(input_file))
    except:
        print(input_file)
#    tpviz.interactive_imshow(viz)

###############################################################################
# load = '/home/phiai/Desktop/samarth/seggro/boxclassifier-tensorpack/FRCNN/try1/output50/maskrcnn/checkpoint' #loading checkpoint
# load = '/home/phiai/Desktop/samarth/seggro/boxclassifier-tensorpack/FRCNN/try1/tensorpack-master/examples/FasterRCNN/train_log_30k/checkpoint'

load = 'model_weights/model-97250.data-00000-of-00001'
register_coco(cfg.DATA.BASEDIR)  # add COCO datasets to the registry
register_balloon(cfg.DATA.BASEDIR)

MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

if not tf.test.is_gpu_available():
    from tensorflow.python.framework import test_util
    assert get_tf_version_tuple() >= (1, 7) and test_util.IsMklEnabled(), \
        "Inference requires either GPU support or MKL support!"

finalize_configs(is_training=False)

cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

predcfg = PredictConfig(
    model=MODEL,
    session_init=SmartInit(load),
    input_names=MODEL.get_inference_tensor_names()[0],
    output_names=MODEL.get_inference_tensor_names()[1])

predictor = OfflinePredictor(predcfg)

test_images = os.listdir('test_images')[:100]

for input_file in test_images: #input_file = test_images[0]
    do_predict(predictor, input_file)
