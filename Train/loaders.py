import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    RandFlipd,
    RandCropByPosNegLabeld,
    SpatialPadd,
    RandAffined,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandSimulateLowResolutiond,

)

import torch
from monai.data import CacheDataset,DataLoader,load_decathlon_datalist, Dataset, PersistentDataset, ThreadDataLoader
from torch.utils.data import DistributedSampler
from TransformswithMONAI import *
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size

def make_CacheCVloaders(transpose_forward,target_spacing,foreground_intensity_properties_per_channel,
                        use_mask_for_norm,is_cascaded,foreground_labels,regions,ignore_label,deep_supervision_scales,
                        patch_size=(96,96,96), samples=1, num_workers=4, batch_size=2, oversample_foreground_percent=0.3,
                        cache_rate=0.1,fold_idx=0,dataset_jsondir= "../dataset_json_list/all_landmark_voxel05_sigma10",seed=42):
    
    
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size):
        """
        This function is stupid and certainly one of the weakest spots of this implementation. Not entirely sure how we can fix it.
        """
        dim = len(patch_size)
        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # todo revisit this parametrization
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            mirror_axes = (0, 1)
        elif dim == 3:
            # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > 3
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
        #  old nnunet for now)
        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            (0.85, 1.25))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]


        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
    
    rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)
    
    """
    print(f"type:{type(initial_patch_size)}")
    initial_patch_size = initial_patch_size.tolist()
    if len(initial_patch_size)==3:
        print("ififif")
        initial_patch_size = [1]+initial_patch_size
    
    print(f"?????:{rotation_for_DA}")
    """
    foreground_intensity_properties_per_channel = foreground_intensity_properties_per_channel["0"]
    percentile_00_5 = foreground_intensity_properties_per_channel["percentile_00_5"]
    percentile_99_5 = foreground_intensity_properties_per_channel["percentile_99_5"]
    mean = foreground_intensity_properties_per_channel["mean"]
    std = foreground_intensity_properties_per_channel["std"]
    
    
    train_transforms =[
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys=["image", "label"]), #<class 'monai.data.meta_tensor.MetaTensor'>
                    TransposeCropnonzerod(keys=["image", "label"],transpose_forward=transpose_forward),
                    nnUNetResampling(keys=["image","label"],mode=["trilinear","nearest"],target_spacing=target_spacing),
                    CTNormalizationd(keys=["image"],percentile_00_5=percentile_00_5,percentile_99_5=percentile_99_5,mean=mean,std=std),
                    SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
                    RandCropByPosLabelForcedd(
                        keys=["image", "label"],
                        label_key="label",
                        spatial_size=patch_size,
                        pos=oversample_foreground_percent,
                        neg=1-oversample_foreground_percent,
                        num_samples=samples,
                        image_key="image",
                        image_threshold=0,
                    ),
                    
                    RandAffined(
                        keys=["image", "label"],  # 画像とラベルを同じ変換で扱う
                        rotate_range=rotation_for_DA,  # 回転の範囲
                        scale_range=(0.7, 1.4),  # スケーリングの範囲
                        prob=0.2,  # 回転・スケーリングの確率
                        padding_mode="zeros",  # パディングはゼロ埋め
                        spatial_size=patch_size,  # パッチサイズ
                        mode=("bilinear", "nearest"),  # 画像はバイリニア補間、ラベルはNearest補間
                    ),
                    RandGaussianNoised(
                        keys=["image"],
                        prob=0.1,
                        mean=0,
                        std=0.1,
                        sample_std=True
                    ),
                    RandGaussianSmoothd(
                        keys=["image"],
                        sigma_x=(0.5, 1.),
                        sigma_y=(0.5, 1.),
                        sigma_z=(0.5, 1.),
                        prob=0.2
                    ),
                    RandMultiplicativeBrightnessTransformdd(
                        keys=["image"],
                        multiplier_range=(0.75, 1.25),
                        synchronize_channels=False,
                        p_per_channel=1,
                        prob=0.15
                    ),
                    RandScaleIntensityd(
                        keys=["image"],        # 変換を適用するキー（画像）
                        factors=(0.75, 1.25),  # ピクセル強度を変更する範囲（倍率の範囲）
                        prob=0.15               # トランスフォームを適用する確率
                    ),
                    RandSimulateLowResolutiond(
                        keys=["image"],
                        prob=0.25,
                        zoom_range=(0.5,1.0)
                    ),
                    RandAdjustContrastd(
                        keys=["image"],
                        prob=0.1,
                        gamma=(0.7, 1.5),
                        invert_image=True,
                        retain_stats=True
                    ),
                    RandAdjustContrastd(
                        keys=["image"],
                        prob=0.3,
                        gamma=(0.7, 1.5),
                        invert_image=False,
                        retain_stats=True
                    )
                    
                    ]
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]), #<class 'monai.data.meta_tensor.MetaTensor'>
        TransposeCropnonzerod(keys=["image", "label"],transpose_forward=transpose_forward),
        nnUNetResampling(keys=["image","label"],mode=["trilinear","nearest"],target_spacing=target_spacing),
        CTNormalizationd(keys=["image"],percentile_00_5=percentile_00_5,percentile_99_5=percentile_99_5,mean=mean,std=std),
        MaskImageTransformd(keys=["label"],set_to=0)
    ])
    
    
    if mirror_axes is not None and len(mirror_axes) > 0:
        train_transforms.append(MirrorTransformd(
                                       keys=["image", "label"],
                                       allowed_axes=mirror_axes
                                ))
    
    if use_mask_for_norm is not None and any(use_mask_for_norm):
        train_transforms.append(MaskImageTransformd(
                                       keys=["image"],
                                       set_to=0
                                   
                                ))
    
    train_transforms.append(MaskImageTransformd(
                                       keys=["label"],
                                       set_to=0))
    print(f"is_cascade:{is_cascaded}")
    if is_cascaded:
        assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
        
        train_transforms= train_transforms +[MoveSegAsOneHotToDataTransformd(
                                    keys=["image"],
                                    source_channel_idx=1,
                                    all_labels=foreground_labels,
                                    remove_channel_from_source=True
                                  ),
                                  RandApplyRandomBinaryOperatorTransformd(
                                        keys=["image"],
                                        channel_idx=list(range(-len(foreground_labels), 0)),
                                        prob=0.4,
                                        strel_size=(1, 8),
                                        p_per_label=1
                                    ),
                                  RandRemoveRandomConnectedComponentFromOneHotEncodingTransformd(
                                        keys=["image"],
                                        channel_idx=list(range(-len(foreground_labels), 0)),
                                        prob=0.4,
                                        fill_with_other_class_p=0,
                                        dont_do_if_covers_more_than_x_percent=0.15,
                                        p_per_label=1
                                    )]
        
        #val_transforms = Compose(list(val_transforms.transforms)
        #                        +[MoveSegAsOneHotToDataTransformd(
        #                            keys=["image"],
        #                            source_channel_idx=1,
        #                            all_labels=foreground_labels,
        #                            remove_channel_from_source=True
        #                          )])
    print(regions)                              
    if regions is not None:
        # the ignore label must also be converted
        
        train_transforms.append(RandConvertSegmentationToRegionsTransformd(
                                    keys=["label"],
                                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                                    channel_in_seg=0
                                ))
        
        val_transforms = Compose(list(val_transforms.transforms)
                                +[RandConvertSegmentationToRegionsTransformd(
                                   keys=["label"],
                                  regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                                  channel_in_seg=0
                            )])
        


    if deep_supervision_scales is not None:
        train_transforms.append(DownsampleSegForDSTransformd(keys=["label"],ds_scales=deep_supervision_scales))
    
    print(f"train trans{len(train_transforms)}")
    train_transforms = Compose(train_transforms)
    
    json_list = os.listdir(dataset_jsondir)
    
    
    file_path = json_list[fold_idx]    
    torch.manual_seed(seed)
    dataset_path = os.path.join(dataset_jsondir,file_path)

    train_files = load_decathlon_datalist(dataset_path, True, "training")
    val_files = load_decathlon_datalist(dataset_path, True, "validation")
    print(f"train case{len(train_files)} val{len(val_files)}")
    print(f"data:{dataset_path}")
        
    train_ds = CacheDataset(data=train_files, transform=train_transforms,cache_rate=cache_rate)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    train_loader = ThreadDataLoader(train_ds, num_workers=num_workers, batch_size=batch_size, shuffle=True,pin_memory=True)
    val_loader = ThreadDataLoader(val_ds, num_workers=num_workers, batch_size=1,pin_memory=True)
        
        
    return train_ds, val_ds, train_loader, val_loader

