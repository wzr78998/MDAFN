"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS


def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.load_pre, args.resume]), \
        'Only support from_scrach or resume or load_preat one time'

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)
    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()
    

if __name__ == '__main__':
    # 创建一个类似于argparse解析出的参数对象
    class Args:
        def __init__(self):
            self.config = 'MDAFN/configs/MDAFN/MDAFN.yml'
            self.resume = None
            self.load_pre= 'MDAFN/tools/pretrained.pth'
            self.device = 'cuda:0'
            self.seed = 42
            self.use_amp = True
            self.output_dir = './output'
            self.summary_dir = None
            self.test_only = False
            self.update = []
            self.print_method = 'builtin'
            self.print_rank = 0
            self.local_rank = None
            
    args = Args()
    main(args)
