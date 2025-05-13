"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import sys
import math
import time
from typing import Iterable

import torch
import torch.amp 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def sinkhorn_distance(a: torch.Tensor, b: torch.Tensor, reg: float = 0.1, n_iters: int = 20) -> torch.Tensor:
    """
    计算两个注意力图之间的Sinkhorn近似EMD距离
    a, b: [B, H, W] 的分布
    返回平均EMD损失标量
    """
    B, H, W = a.shape
    a_flat = a.view(B, -1)
    b_flat = b.view(B, -1)
    N = H * W
    # 构建坐标网格
    coords = torch.stack(torch.meshgrid(torch.arange(H, device=a.device), torch.arange(W, device=a.device), indexing='ij'), dim=-1)
    coords = coords.view(N, 2).float()
    # 计算成本矩阵
    cost = torch.cdist(coords, coords, p=2)
    M = cost / (cost.max() + 1e-8)
    # Sinkhorn迭代
    K = torch.exp(-M / reg)
    u = torch.full((B, N), 1.0 / N, device=a.device)
    v = torch.full((B, N), 1.0 / N, device=a.device)
    for _ in range(n_iters):
        u = a_flat / (K @ v.T).T.clamp(min=1e-8)
        v = b_flat / (K.T @ u.T).T.clamp(min=1e-8)
    # 运输矩阵
    P = u.unsqueeze(2) * K.unsqueeze(0) * v.unsqueeze(1)
    emd = torch.sum(P * M.unsqueeze(0), dim=(1, 2))
    return emd.mean()


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples_vis = samples['vis'].to(device)
        samples_ir = samples['ir'].to(device)
        samples=torch.cat((samples_vis,samples_ir),dim=1)
        del samples_vis,samples_ir
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)
            
            # 处理熵正则损失
            if isinstance(outputs, dict) and 'entropy_loss' in outputs:
                loss_dict['entropy_loss'] = outputs['entropy_loss']
            
            # 添加EMD损失，用于对齐两模态的注意力图
            if isinstance(outputs, dict) and 'vis_attention' in outputs and 'ir_attention' in outputs:
                # 使用模型前向输出的动态EMD权重，仅训练阶段提供
                if 'emd_weights' in outputs:
                    weights = outputs['emd_weights']
                    w0 = weights[:, 0].mean()
                    w1 = weights[:, 1].mean()
                    loss_dict['loss_emd'] = w0 * sinkhorn_distance(outputs['vis_attention'], outputs['ir_attention']) + \
                                            w1 * sinkhorn_distance(outputs['ir_attention'], outputs['vis_attention'])
                    # 记录权重以便监控
                    loss_dict['emd_weight1'] = w0
                    loss_dict['emd_weight2'] = w1
                else:
                    # 使用默认均等权重
                    loss_dict['loss_emd'] = sinkhorn_distance(outputs['vis_attention'], outputs['ir_attention']) + \
                                            sinkhorn_distance(outputs['ir_attention'], outputs['vis_attention'])

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)
            
            # 处理熵正则损失
            if isinstance(outputs, dict) and 'entropy_loss' in outputs:
                loss_dict['entropy_loss'] = outputs['entropy_loss']
            
            # 添加EMD损失，用于对齐两模态的注意力图
            if isinstance(outputs, dict) and 'vis_attention' in outputs and 'ir_attention' in outputs:
                # 使用模型前向输出的动态EMD权重，仅训练阶段提供
                if 'emd_weights' in outputs:
                    weights = outputs['emd_weights']
                    w0 = weights[:, 0].mean()
                    w1 = weights[:, 1].mean()
                    loss_dict['loss_emd'] =0.1*( w0 * sinkhorn_distance(outputs['vis_attention'], outputs['ir_attention']) + \
                                            w1 * sinkhorn_distance(outputs['ir_attention'], outputs['vis_attention']))
                    # 记录权重以便监控
                    loss_dict['emd_weight1'] = w0
                    loss_dict['emd_weight2'] = w1
                else:
                    # 使用默认均等权重
                    loss_dict['loss_emd'] = 0.1*(sinkhorn_distance(outputs['vis_attention'], outputs['ir_attention']) + \
                                            sinkhorn_distance(outputs['ir_attention'], outputs['vis_attention']))
            
            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

   
        emd_loss = None
        if 'loss_emd' in loss_dict:
            emd_loss = loss_dict.pop('loss_emd', None)
            
        entropy_loss = None
        if 'entropy_loss' in loss_dict:
            entropy_loss = loss_dict.pop('entropy_loss', None)
            
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        # 恢复loss_dict
      
       
     

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # 只更新总损失，不更新分解的损失
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            # 不记录分解的损失
            # for k, v in loss_dict_reduced.items():
            #     writer.add_scalar(f'Loss/{k}', v.item(), global_step)
                
            # # 记录熵正则损失
            # if entropy_loss is not None:
            #     writer.add_scalar('Loss/entropy_loss', entropy_loss.item(), global_step)
                    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()
    iou_types = coco_evaluator.iou_types

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # 添加FPS测量
    total_fps = 0
    frame_count = 0
    
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples_vis = samples['vis'].to(device)
        samples_ir = samples['ir'].to(device)
        samples=torch.cat((samples_vis,samples_ir),dim=1)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 测量推理时间
        start_time = time.time()
        outputs = model(samples)
        torch.cuda.synchronize()  # 确保GPU操作完成
        inference_time = time.time() - start_time
        
        # 计算每批次的FPS
        batch_size = samples_vis.shape[0]
        fps = batch_size / inference_time
        total_fps += fps
        frame_count += 1

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        results = postprocessor(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # 计算平均FPS
    avg_fps = total_fps / frame_count if frame_count > 0 else 0
    print(f"\nAverage FPS: {avg_fps:.2f}")
    
    # 加入FPS到metrics中
    metric_logger.meters['fps'] = SmoothedValue(window_size=1, fmt='{global_avg:.2f}')
    metric_logger.update(fps=avg_fps)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # 加入FPS到输出统计
    stats['fps'] = avg_fps
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    return stats, coco_evaluator



