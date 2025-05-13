"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
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
class M3FDDetection(DetDataset):
    __inject__ = ['transforms', ]

    def __init__(self, root: str, ann_file: str = "trainval.txt", label_file: str = "label_list.txt", transforms: Optional[Callable] = None):
        """
        M3FD多模态(可见光和红外)数据集加载类
        
        Args:
            root: 数据集ImageSets/Main目录路径
            ann_file: 包含图像ID的文本文件
            label_file: 标签列表文件
            transforms: 数据增强和预处理
        """
        with open(os.path.join(root, ann_file), 'r') as f:
            lines = [x.strip() for x in f.readlines()]
            
        # 经过分析，路径中可能包含多个ImageSets层级，我们需要找到M3FD的基础目录
        # root路径示例: '/home/skj/RT-DETR-main (1)/RT-DETR-main/MDAFNv2_pytorch/dataset/M3FD/ImageSets/Main'
        
        # 首先处理可能的路径切分
        parts = root.split(os.sep)
        
        # 找到ImageSets在路径中的位置
        try:
            imagesets_idx = parts.index('ImageSets')
            # 取ImageSets之前的所有路径作为基础目录
            base_parts = parts[:imagesets_idx]
            m3fd_base_dir = os.sep.join(base_parts)
        except ValueError:
            # 如果没有找到ImageSets，使用默认方法
            m3fd_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(root)))
        
        # 确保基础目录存在
        if not os.path.exists(m3fd_base_dir):
            raise ValueError(f"无法找到有效的基础目录: {m3fd_base_dir}")
        
        # 现在所有目录都是同级的
        vis_jpeg_dir = os.path.join(m3fd_base_dir, "JPEGImages")  # 可见光图像目录
        ir_jpeg_dir = os.path.join(m3fd_base_dir, "JPEGImages_ir")  # 红外图像目录
        anno_dir = os.path.join(m3fd_base_dir, "Annotations")
        
        # 输出调试信息，确认路径是否正确
        print(f"基础目录: {m3fd_base_dir}")
        print(f"VIS目录: {vis_jpeg_dir}")
        print(f"IR目录: {ir_jpeg_dir}")
        print(f"标注目录: {anno_dir}")
        
        # 验证目录是否存在
        if not all(os.path.exists(d) for d in [vis_jpeg_dir, ir_jpeg_dir, anno_dir]):
            print(f"警告: 某些目录不存在!")
            print(f"VIS目录存在: {os.path.exists(vis_jpeg_dir)}")
            print(f"IR目录存在: {os.path.exists(ir_jpeg_dir)}")
            print(f"标注目录存在: {os.path.exists(anno_dir)}")
        
        # 存储图像和标注文件路径
        self.vis_images = [os.path.join(vis_jpeg_dir, f"{line}.png") for line in lines]
        self.ir_images = [os.path.join(ir_jpeg_dir, f"{line}.png") for line in lines]
        self.targets = [os.path.join(anno_dir, f"{line}.xml") for line in lines]
        
        # 确保所有文件都存在且数量一致
        assert len(self.vis_images) == len(self.ir_images) == len(self.targets)
        
        # 加载标签映射
        label_path = os.path.join(m3fd_base_dir, label_file) if not os.path.exists(os.path.join(root, label_file)) else os.path.join(root, label_file)
        with open(label_path, 'r') as f:
            labels = f.readlines()
            labels = [lab.strip() for lab in labels]

        self.transforms = transforms
        self.labels_map = {lab: i for i, lab in enumerate(labels)}
        
    def __len__(self):
        return len(self.vis_images)
        
    def __getitem__(self, index: int):
        multi_images, target = self.load_item(index)
        
        if self.transforms is not None:
            # 对两个模态分别应用相同的变换，确保对齐
            vis_image, target0, _ = self.transforms(multi_images['vis'], target, self)
            ir_image, _, _ = self.transforms(multi_images['ir'], target, self)
            multi_images = {'vis': vis_image, 'ir': ir_image}
            
        return multi_images, target0
    def load_item(self, index: int):
        """加载一对图像(可见光和红外)和对应的标注"""
        # 加载两个模态的图像
        vis_image = Image.open(self.vis_images[index]).convert("RGB")
        ir_image = Image.open(self.ir_images[index]).convert("RGB")  # 红外图像也转为RGB格式方便处理
        
        # 确保两个图像大小一致
        assert vis_image.size == ir_image.size, f"图像大小不一致: vis={vis_image.size}, ir={ir_image.size}"
        
        # 解析XML标注，使用ElementTree直接获取对象列表
        xml_root = ET_parse(self.targets[index]).getroot()
        objects = xml_root.findall('object')
        
        # 构建输出格式
        output = {}
        output["image_id"] = torch.tensor([index])
        for k in ['area', 'boxes', 'labels', 'iscrowd']:
            output[k] = []
            
        for obj in objects:
            # 获取边框坐标
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            box = [xmin, ymin, xmax, ymax]
            output["boxes"].append(box)
            # 获取标签名称
            label_name = obj.find('name').text
            output["labels"].append(label_name)
            # 计算面积
            output["area"].append((xmax - xmin) * (ymax - ymin))
            output["iscrowd"].append(0)

        w, h = vis_image.size
        boxes = torch.tensor(output["boxes"]) if len(output["boxes"]) > 0 else torch.zeros(0, 4)
        output['boxes'] = convert_to_tv_tensor(boxes, 'boxes', box_format='xyxy', spatial_size=[h, w])
        output['labels'] = torch.tensor([self.labels_map[lab] for lab in output["labels"]])
        output['area'] = torch.tensor(output['area'])
        output["iscrowd"] = torch.tensor(output["iscrowd"])
        output["orig_size"] = torch.tensor([w, h])
        
        # 返回两个模态的图像字典和标注
        return {'vis': vis_image, 'ir': ir_image}, output
    
    def parse_voc_xml(self, node):
        """解析VOC XML格式的标注"""
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = {}
            for dc in children:
                dc_children = list(dc)
                if dc_children:
                    dc_dict = {}
                    for dcc in dc_children:
                        dcc_children = list(dcc)
                        if dcc_children:
                            dcc_dict = {}
                            for dccc in dcc_children:
                                dcc_dict[dccc.tag] = dccc.text
                            dc_dict[dcc.tag] = dcc_dict
                        else:
                            dc_dict[dcc.tag] = dcc.text
                    def_dic[dc.tag] = dc_dict
                else:
                    def_dic[dc.tag] = dc.text
            voc_dict[node.tag] = def_dic
        else:
            voc_dict[node.tag] = node.text
        return voc_dict 