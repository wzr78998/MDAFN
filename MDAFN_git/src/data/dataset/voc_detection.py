"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from sympy import im
import torch
import torchvision
import torchvision.transforms.functional as TVF 

import os
from PIL import Image
from typing import Optional, Callable

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register

@register()
class VOCDetection(torchvision.datasets.VOCDetection, DetDataset):
    __inject__ = ['transforms', ]

    def __init__(self, root: str, ann_file: str = "trainval.txt", label_file: str = "label_list.txt", transforms: Optional[Callable] = None):

        with open(os.path.join(root, ann_file), 'r') as f:
            lines = [x.strip() for x in f.readlines()]
            
        # 修改：兼容只有ID的格式
        base_dir = os.path.dirname(os.path.dirname(root))  # 假设root是ImageSets/Main，上两级是数据集根目录
        jpeg_dir = os.path.join(base_dir, "JPEGImages")
        anno_dir = os.path.join(base_dir, "Annotations")
        
        self.images = [os.path.join(jpeg_dir, f"{line}.png") for line in lines]
        self.targets = [os.path.join(anno_dir, f"{line}.xml") for line in lines]  # 修改为targets
        assert len(self.images) == len(self.targets)

        label_path = os.path.join(base_dir, label_file) if not os.path.exists(os.path.join(root, label_file)) else os.path.join(root, label_file)
        with open(label_path, 'r') as f:
            labels = f.readlines()
            labels = [lab.strip() for lab in labels]

        self.transforms = transforms
        self.labels_map = {lab: i for i, lab in enumerate(labels)}
        
    def __getitem__(self, index: int):
        image, target = self.load_item(index)
        if self.transforms is not None:
            image, target, _ = self.transforms(image, target, self)        
        # target["orig_size"] = torch.tensor(TVF.get_image_size(image))
        return image, target

    def load_item(self, index: int):
        image = Image.open(self.images[index]).convert("RGB")
        target = self.parse_voc_xml(ET_parse(self.targets[index]).getroot())  # 使用self.targets
        
        output = {}
        output["image_id"] = torch.tensor([index])
        for k in ['area', 'boxes', 'labels', 'iscrowd']:
            output[k] = []
            
        for blob in target['annotation']['object']:
            box = [float(v) for v in blob['bndbox'].values()]
            output["boxes"].append(box)
            output["labels"].append(blob['name'])
            output["area"].append((box[2] - box[0]) * (box[3] - box[1]))
            output["iscrowd"].append(0)

        w, h = image.size
        boxes = torch.tensor(output["boxes"]) if len(output["boxes"]) > 0 else torch.zeros(0, 4)
        output['boxes'] = convert_to_tv_tensor(boxes, 'boxes', box_format='xyxy', spatial_size=[h, w])
        output['labels'] = torch.tensor([self.labels_map[lab] for lab in output["labels"]])
        output['area'] = torch.tensor(output['area'])
        output["iscrowd"] = torch.tensor(output["iscrowd"])
        output["orig_size"] = torch.tensor([w, h])
        
        return image, output
    
