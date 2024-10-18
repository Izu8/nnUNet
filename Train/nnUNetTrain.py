import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Tuple, Union, List

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json


from lightning.fabric import Fabric
from loaders import make_CacheCVloaders
from build_loss import build_loss
from nnunetv2_ import get_network
from TrainLoop import CVLoop
import argparse

if __name__ == "__main__":
    

    # nnU-Netの環境変数を設定
    os.environ['nnUNet_raw'] = "/win/scallop/user/ogura/nnUNet/datasets/nnUnet_raw"
    os.environ['nnUNet_preprocessed'] = "/win/scallop/user/ogura/nnUNet/datasets/nnUnet_preprocessed"
    os.environ['nnUNet_results'] = "/win/scallop/user/ogura/nnUNet/datasets/datasets/nnUnet_results"

    # 環境変数が正しく設定されたか確認
    print('nnUNet_raw_data_base:', os.environ.get('nnUNet_raw'))
    print('nnUNet_preprocessed:', os.environ.get('nnUNet_preprocessed'))
    print('RESULTS_FOLDER:', os.environ.get('nnUNet_results'))
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
    from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian
    from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
    from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
    from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
    from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
    from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
    from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
    from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
    from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
    from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
    from nnunetv2.utilities.collate_outputs import collate_outputs
    from nnunetv2.utilities.crossval_split import generate_crossval_split
    from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
    from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.helpers import empty_cache, dummy_context
    from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument('--dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('--configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('--plan', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    #parser.add_argument("--run_name", default="nnUnet_scratch", type=str, help="run name")
    parser.add_argument("--infer_device", default="cpu", type=str, help="device for sliding window inference")
    parser.add_argument("--num_workers", default=8, type=int, help="number of workers")
    parser.add_argument("--dataset_jsondir", default="/win/scallop/user/ogura/nnUNet/dataset_json_list/Dataset012_Swallowing-8structures_faststor", type=str, help="dataset json dir")
    parser.add_argument("--parallel", action='store_false', help="multi gpu processing or not")
    parser.add_argument("--val_interval", default=100, type=int, help="validation interval")
    parser.add_argument("--cache_rate",default=0, type=float,help="persistent dataset")
    parser.add_argument("--resume",default=0, type=int, help="resume epoch")
    parser.add_argument("--fold", nargs="*", type=int, help="fold")
    #parser.add_argument("--train_interval", default=100, type=int, help="train interval")
    args = parser.parse_args()
    
    
    for count, fold_idx in enumerate(args.fold):
        
    
        preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(args.dataset_name_or_id))
        plans_file = join(preprocessed_dataset_folder_base, args.plan + f'fold{fold_idx}.json')
        plans = load_json(plans_file)
        PLANS_MANAGER = PlansManager(plans)
        CONFIG_MANAGER = PLANS_MANAGER.get_configuration(args.configuration)
        DATASET_JSON = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
        label_manager = PLANS_MANAGER.get_label_manager(DATASET_JSON)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        RUN_NAME = "nnUNet_"+args.configuration
        
        print(f"plans_file:{plans_file}")
                
        output_folder_base = join(nnUNet_results, PLANS_MANAGER.dataset_name,
                                '__' + PLANS_MANAGER.plans_name + "__" + args.configuration) \
                if nnUNet_results is not None else None
                
        output_folder = join(output_folder_base, f'fold{fold_idx}')
        print(f"out folder:{output_folder}")

        preprocessed_dataset_folder = join(preprocessed_dataset_folder_base, CONFIG_MANAGER.data_identifier)
        
        is_cascaded = CONFIG_MANAGER.previous_stage_name is not None
        folder_with_segs_from_previous_stage = \
                join(nnUNet_results, PLANS_MANAGER.dataset_name,
                    '__' + PLANS_MANAGER.plans_name + "__" +
                    CONFIG_MANAGER.previous_stage_name, 'predicted_next_stage', args.configuration) \
                    if is_cascaded else None
        
        fabric = Fabric(accelerator="gpu", devices="auto")
        fabric.launch()
        
        ### Some hyperparameters for you to fiddle with
        initial_lr = 1e-2
        weight_decay = 3e-5
        oversample_foreground_percent = 0.33
        num_iterations_per_epoch = 250
        num_val_iterations_per_epoch = 50
        num_epochs = 1000
        
        if args.resume == 0 and count == 0:
            current_epoch = 0
        else:
            current_epoch = args.resume
        
        
        enable_deep_supervision = True
        if enable_deep_supervision:
                deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(CONFIG_MANAGER.pool_op_kernel_sizes), axis=0))[:-1]
        else:
            deep_supervision_scales = None
            
        has_regions = label_manager.foreground_regions if label_manager.has_regions else None
        ignore_label = label_manager.ignore_label
        world_size = 1
        out_channels = label_manager.num_segmentation_heads
        
        num_input_channels =  determine_num_input_channels(PLANS_MANAGER, CONFIG_MANAGER,DATASET_JSON)
        model =   get_network(num_input_channels, out_channels, CONFIG_MANAGER.spacing, CONFIG_MANAGER.patch_size,enable_deep_supervision)
        #model = model.to(DEVICE)
        optimizer =torch.optim.SGD(model.parameters(), initial_lr, weight_decay=weight_decay,
                                momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, initial_lr, num_epochs)
        #scaler = torch.cuda.amp.GradScaler()
        best_val_score = None
        
        if args.resume == 0 and count == 0:
            pass
        else:
            print(f"loading epoch{current_epoch}model")
            checkpoint = torch.load(f"../net_params/{PLANS_MANAGER.dataset_name}/{RUN_NAME+'_'+args.plan}/fold{fold_idx}/{RUN_NAME + '_fold' + str(fold_idx)}_epoch{current_epoch}model.pth")
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
            #scaler.load_state_dict(checkpoint["grad scaler"])
            best_val_score = checkpoint["best val score"]
                
                
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(DEVICE)
        
        
        loss = build_loss(CONFIG_MANAGER.batch_dice,has_regions,ignore_label,enable_deep_supervision,deep_supervision_scales,world_size)
        
        
        train_ds, val_ds, train_loader, val_loader  = make_CacheCVloaders(PLANS_MANAGER.transpose_forward,CONFIG_MANAGER.spacing,PLANS_MANAGER.foreground_intensity_properties_per_channel,
                                                                        CONFIG_MANAGER.use_mask_for_norm, is_cascaded=is_cascaded,
                                                                        foreground_labels=label_manager.foreground_labels,regions=has_regions,
                                                                        ignore_label=ignore_label,deep_supervision_scales=deep_supervision_scales,
                                                                        patch_size=CONFIG_MANAGER.patch_size, 
                                                                        samples=1, num_workers=args.num_workers, batch_size=CONFIG_MANAGER.batch_size, 
                                                                        oversample_foreground_percent=oversample_foreground_percent,
                                                                        cache_rate=args.cache_rate,fold_idx=fold_idx,dataset_jsondir=args.dataset_jsondir,seed=args.seed)
        
        
        config = {
 
                    "seed":args.seed,
                    "plan":args.plan,
                    "num_workers": args.num_workers,
                    "label_names":label_manager.label_dict,
                    "cache_rate":args.cache_rate,

                    # train settings
                    "train_batch_size": CONFIG_MANAGER.batch_size,
                    "val_batch_size": 1,
                    "lr": initial_lr,
                    "max_epochs": num_epochs,
                    "rand_crop_num":1,
                    "voxel_size":CONFIG_MANAGER.spacing,
                    "sw_batch_size":CONFIG_MANAGER.batch_size,
                    "val_interval":args.val_interval,
                    "parallel":"",
                    "infer_device":args.infer_device,
                    "pretrain":"",
                    #"train_interval":args.train_interval,
                    #"lr_scheduler": "cosine_decay", # just to keep track




                    # Unet model (you can even use nested dictionary and this will be handled by W&B automatically)
                    "model_type": "nnUNet_"+args.configuration, # just to keep track
                    "model_params": dict(
                        image_size=CONFIG_MANAGER.patch_size,
                        feature_size=48),
                    "class_num":len(label_manager.label_dict),
                }
        print(f"config:{config}")
        print(list(config["label_names"].keys())[1:])
        
        
        model, optimizer = fabric.setup(model, optimizer)
        train_loader, val_loader = fabric.setup_dataloaders(train_loader,val_loader)
        
        Trainer = CVLoop(model,config,current_epoch,num_epochs,num_iterations_per_epoch,train_loader,val_loader,loss,optimizer,lr_scheduler,
                        fabric,RUN_NAME,args.infer_device,CONFIG_MANAGER.batch_size,fold_idx,args.val_interval,best_val_score=best_val_score,result_dir=PLANS_MANAGER.dataset_name,
                        net_param_dir=PLANS_MANAGER.dataset_name,summary_path="summary_8structures.csv")
        Trainer.train()