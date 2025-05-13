"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib

# MiXaiLL76 replacing pycocotools with faster-coco-eval for better performance and support.
"""

import os
import contextlib
import copy
import numpy as np
import torch

from faster_coco_eval import COCO, COCOeval_faster
import faster_coco_eval.core.mask as mask_util
from ...core import register
from ...misc import dist_utils
__all__ = ['CocoEvaluator',]


# 添加一个调试转换过程的函数
def debug_box_conversion(boxes, conversion_type="xyxy->xywh", reverse=False):
    """调试边界框转换过程
    
    Args:
        boxes: 边界框张量
        conversion_type: 转换类型描述
        reverse: 是否为反向转换
    """
    if boxes.numel() == 0:
        return
        
    print(f"\n===== DEBUG {conversion_type} 转换 {'(反向)' if reverse else ''} =====")
    # 只展示前5个框
    num_boxes_to_show = min(5, boxes.shape[0])
    for i in range(num_boxes_to_show):
        box = boxes[i].tolist()
        print(f"框 {i}: {box}")
        
        # 根据转换类型计算反向或正向结果进行验证
        if conversion_type == "xyxy->xywh":
            if not reverse:
                # xyxy -> xywh
                x1, y1, x2, y2 = box
                xywh = [x1, y1, x2 - x1, y2 - y1]
                print(f"  转换后 xywh: {xywh}")
                # 验证转换回来是否一致
                x, y, w, h = xywh
                xyxy_back = [x, y, x + w, y + h]
                print(f"  还原回 xyxy: {xyxy_back}")
            else:
                # xywh -> xyxy
                x, y, w, h = box
                xyxy = [x, y, x + w, y + h]
                print(f"  转换后 xyxy: {xyxy}")
    
    print("================================\n")
    

@register()
class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt : COCO = coco_gt
        self.iou_types = iou_types

        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(coco_gt, iouType=iou_type, print_function=print, separate_eval=True)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(self.coco_gt, iouType=iou_type, print_function=print, separate_eval=True)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}


    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(np.array(coco_eval._evalImgs_cpp).reshape(len(coco_eval.params.catIds), len(coco_eval.params.areaRng), len(coco_eval.params.imgIds)))

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def convert_to_xywh(self, boxes):
        """将XYXY格式转换为XYWH格式
        
        Args:
            boxes: [N, 4] 格式为 [x1, y1, x2, y2] 的边界框张量
            
        Returns:
            [N, 4] 格式为 [x, y, width, height] 的边界框张量
        """
        # 确保输入是XYXY格式: [x1, y1, x2, y2]
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        
        # 计算宽度和高度
        width = xmax - xmin
        height = ymax - ymin
        
        # 检查是否有非法值
        if (width < 0).any() or (height < 0).any():
            problematic = (width < 0) | (height < 0)
            num_issues = problematic.sum().item()
            print(f"警告: 发现{num_issues}个框具有负宽度或高度!")
            print("问题框示例:")
            indices = torch.where(problematic)[0]
            for i in range(min(5, len(indices))):
                idx = indices[i].item()
                print(f"  框{idx}: [x1={xmin[idx]:.1f}, y1={ymin[idx]:.1f}, x2={xmax[idx]:.1f}, y2={ymax[idx]:.1f}]")
                print(f"  宽度={width[idx]:.1f}, 高度={height[idx]:.1f}")
            
            # 修复问题框 - 确保宽度和高度为正
            width = torch.abs(width)
            height = torch.abs(height)
        
        # 返回COCO格式: [x, y, width, height]
        return torch.stack((xmin, ymin, width, height), dim=1)

    def prepare_for_coco_detection(self, predictions):
        """将检测结果转换为COCO API格式
        
        Args:
            predictions: 每个图像的预测结果字典
            
        Returns:
            coco_results: COCO API需要的格式列表
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            
            # # 调试原始预测框
            # print(f"\n检查图像 {original_id} 的预测框:")
            # print(f"原始预测框类型: {boxes.dtype}, 形状: {boxes.shape}")
            # print(f"值范围: min={boxes.min().item():.4f}, max={boxes.max().item():.4f}")
            # if boxes.shape[0] > 0:
            #     print(f"前3个框示例: {boxes[:min(3, boxes.shape[0])]}")
            
            # # 转换前进行调试
            # debug_box_conversion(boxes, "xyxy->xywh")
            
            # 执行转换 - 从XYXY格式转换为COCO需要的XYWH格式
            boxes = self.convert_to_xywh(boxes)
            
            # # 转换后进行调试
            # if boxes.shape[0] > 0:
            #     print(f"转换后值范围: min={boxes.min().item():.4f}, max={boxes.max().item():.4f}")
            #     print(f"COCO格式框示例: {boxes[:min(3, boxes.shape[0])]}")
            
            # 检查转换结果是否有非法值
            if (boxes[:, 2:] <= 0).any():
                print("警告: 存在宽度或高度小于等于0的框!")
                # 找出有问题的框
                problem_indices = torch.where((boxes[:, 2] <= 0) | (boxes[:, 3] <= 0))[0]
                for idx in problem_indices[:5]:  # 只显示前5个问题框
                    print(f"问题框 {idx}: {boxes[idx].tolist()}, 原始框: {prediction['boxes'][idx].tolist()}")
                
                # 修复这些问题框，强制宽高为正
                boxes[:, 2:] = torch.clamp(boxes[:, 2:], min=1.0)
            
            # 转换为列表
            boxes = boxes.tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = self.convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def merge(img_ids, eval_imgs):
    all_img_ids = dist_utils.all_gather(img_ids)
    all_eval_imgs = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)


    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2).ravel()
    # merged_eval_imgs = np.array(merged_eval_imgs).T.ravel()

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()