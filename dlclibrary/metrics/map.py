"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import itertools
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def calc_object_keypoint_similarity(
    xy_pred,
    xy_true,
    sigma,
    margin=0,
    symmetric_kpts=None,
):
    visible_gt = ~np.isnan(xy_true).all(axis=1)
    if visible_gt.sum() < 2:  # At least 2 points needed to calculate scale
        return np.nan
    true = xy_true[visible_gt]
    scale_squared = np.product(np.ptp(true, axis=0) + np.spacing(1) + margin * 2)
    if np.isclose(scale_squared, 0):
        return np.nan
    k_squared = (2 * sigma) ** 2
    denom = 2 * scale_squared * k_squared
    if symmetric_kpts is None:
        pred = xy_pred[visible_gt]
        pred[np.isnan(pred)] = np.inf
        dist_squared = np.sum((pred - true) ** 2, axis=1)
        oks = np.exp(-dist_squared / denom)
        return np.mean(oks)
    else:
        oks = []
        xy_preds = [xy_pred]
        combos = (
            pair
            for l in range(len(symmetric_kpts))
            for pair in itertools.combinations(symmetric_kpts, l + 1)
        )
        for pairs in combos:
            # Swap corresponding keypoints
            tmp = xy_pred.copy()
            for pair in pairs:
                tmp[pair, :] = tmp[pair[::-1], :]
            xy_preds.append(tmp)
        for xy_pred in xy_preds:
            pred = xy_pred[visible_gt]
            pred[np.isnan(pred)] = np.inf
            dist_squared = np.sum((pred - true) ** 2, axis=1)
            oks.append(np.mean(np.exp(-dist_squared / denom)))
        return max(oks)


def match_assemblies(
    ass_pred,
    ass_true,
    sigma,
    margin=0,
    symmetric_kpts=None,
    greedy_matching=False
):
    # Only consider assemblies of at least two keypoints
    ass_pred = [a for a in ass_pred if len(a) > 1]
    ass_true = [a for a in ass_true if len(a) > 1]

    matched = []

    # Greedy assembly matching like in pycocotools
    if greedy_matching:
        inds_true = list(range(len(ass_true)))
        inds_pred = np.argsort(
            [ins.affinity if ins.n_links else ins.confidence for ins in ass_pred]
        )[::-1]
        for ind_pred in inds_pred:
            xy_pred = ass_pred[ind_pred].xy
            oks = []
            for ind_true in inds_true:
                xy_true = ass_true[ind_true].xy
                oks.append(
                    calc_object_keypoint_similarity(
                        xy_pred, xy_true, sigma, margin, symmetric_kpts,
                    )
                )
            if np.all(np.isnan(oks)):
                continue
            ind_best = np.nanargmax(oks)
            ind_true_best = inds_true.pop(ind_best)
            matched.append((ass_pred[ind_pred], ass_true[ind_true_best], oks[ind_best]))
            if not inds_true:
                break

    # Global rather than greedy assembly matching
    else:
        mat = np.zeros((len(ass_pred), len(ass_true)))
        for i, a_pred in enumerate(ass_pred):
            for j, a_true in enumerate(ass_true):
                oks = calc_object_keypoint_similarity(
                    a_pred.xy, a_true.xy, sigma, margin, symmetric_kpts,
                )
                if ~np.isnan(oks):
                    mat[i, j] = oks
        rows, cols = linear_sum_assignment(mat, maximize=True)
        inds_true = list(range(len(ass_true)))
        for row, col in zip(rows, cols):
            matched.append((ass_pred[row], ass_true[col], mat[row, col]))
            _ = inds_true.remove(col)

    unmatched = [ass_true[ind] for ind in inds_true]
    return matched, unmatched



def evaluate_assembly(
    ass_pred_dict,
    ass_true_dict,
    oks_sigma=0.072,
    oks_thresholds=np.linspace(0.5, 0.95, 10),
    margin=0,
    symmetric_kpts=None,
    greedy_matching=False,
):
    # sigma is taken as the median of all COCO keypoint standard deviations
    all_matched = []
    all_unmatched = []
    for ind, ass_true in tqdm(ass_true_dict.items()):
        ass_pred = ass_pred_dict.get(ind, [])
        matched, unmatched = match_assemblies(
            ass_pred,
            ass_true,
            oks_sigma,
            margin,
            symmetric_kpts,
            greedy_matching,
        )
        all_matched.extend(matched)
        all_unmatched.extend(unmatched)
    if not all_matched:
        return {
            "precisions": np.array([]),
            "recalls": np.array([]),
            "mAP": 0.0,
            "mAR": 0.0,
        }

    conf_pred = np.asarray([match[0].affinity for match in all_matched])
    idx = np.argsort(-conf_pred, kind="mergesort")
    # Sort matching score (OKS) in descending order of assembly affinity
    oks = np.asarray([match[2] for match in all_matched])[idx]
    ntot = len(all_matched) + len(all_unmatched)
    recall_thresholds = np.linspace(0, 1, 101)
    precisions = []
    recalls = []
    for th in oks_thresholds:
        tp = np.cumsum(oks >= th)
        fp = np.cumsum(oks < th)
        rc = tp / ntot
        pr = tp / (fp + tp + np.spacing(1))
        recall = rc[-1]
        # Guarantee precision decreases monotonically
        # See https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)
        for i in range(len(pr) - 1, 0, -1):
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]
        inds_rc = np.searchsorted(rc, recall_thresholds)
        precision = np.zeros(inds_rc.shape)
        valid = inds_rc < len(pr)
        precision[valid] = pr[inds_rc[valid]]
        precisions.append(precision)
        recalls.append(recall)
    precisions = np.asarray(precisions)
    recalls = np.asarray(recalls)
    return {
        "precisions": precisions,
        "recalls": recalls,
        "mAP": precisions.mean(),
        "mAR": recalls.mean(),
    }
