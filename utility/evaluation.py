import numpy as np
import os
import six
from .image_util import *

n_classes = 20
    
def calc_semantic_segmentation_confusion(pred_labels, gt_labels):

    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    n_class = 21
    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')
        pred_label = pred_label.flatten()
        gt_label = gt_label.flatten()

        # Count statistics from valid pixels.
        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) +
            pred_label[mask], minlength=n_class**2).reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')
    return confusion

def calc_depth_confusion(pred_labels, gt_labels, depth_maps):

    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    n_class = 21
    depth_stage = 10
    
    confusion = np.zeros((depth_stage, n_class, n_class), dtype=np.int64)
    for pred_label, gt_label, depth_map in six.moves.zip(pred_labels, gt_labels, depth_maps):
        if pred_label.ndim != 2 or gt_label.ndim != 2 or depth_map.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_label.shape != gt_label.shape or pred_label.shape != depth_map.shape:
            raise ValueError('Shape of ground truth and prediction and depth_map should'
                             ' be same.')
        mean_depth_map = mean_depth(depth_map, gt_label)
        
        pred_label = pred_label.flatten()
        gt_label = gt_label.flatten()
        depth = mean_depth_map.flatten()
        
        # Count statistics from valid pixels.
        mask = depth >= 0
        confusion += np.bincount(
            n_class**2 * depth[mask] + n_class *gt_label[mask]
            + pred_label[mask], minlength=n_class**2 * depth_stage).reshape((depth_stage, n_class, n_class))
        

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')
    return confusion
    
        
    
        