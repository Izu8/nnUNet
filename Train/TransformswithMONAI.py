from time import time
from typing import Union, List, Tuple, Callable

import numpy as np
import torch
from fft_conv_pytorch import fft_conv

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform
from skimage.morphology import ball, disk
from skimage.morphology.binary import binary_erosion, binary_dilation, binary_closing, binary_opening
from acvl_utils.morphology.morphology_helper import label_with_component_sizes

from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
import torch.nn.functional as F
from monai.transforms import MapTransform, RandomizableTransform, Resize, SpatialCrop
from torch.nn.functional import interpolate


import warnings
from collections.abc import Callable, Sequence, Hashable, Mapping
from copy import deepcopy

import numpy as np
import torch

from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.transforms.croppad.functional import crop_func, pad_func
from monai.transforms.inverse import InvertibleTransform, TraceableTransform
from monai.transforms.traits import MultiSampleTrait
from monai.transforms.transform import LazyTransform, Randomizable, Transform
from monai.transforms.utils import (
    compute_divisible_spatial_size,
    generate_label_classes_crop_centers,
    generate_pos_neg_label_crop_centers,
    correct_crop_centers,
    generate_spatial_bounding_box,
    is_positive,
    map_binary_to_indices,
    map_classes_to_indices,
    weighted_patch_samples,
)
from monai.transforms.utils_pytorch_numpy_unification import unravel_index
from monai.utils import ImageMetaKey as Key
from monai.utils import (
    LazyAttr,
    Method,
    PytorchPadMode,
    TraceKeys,
    TransformBackends,
    convert_data_type,
    convert_to_tensor,
    deprecated_arg_default,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
    pytorch_after,
)


def binary_dilation_torch(input_tensor, structure_element):
    # Convert the boolean tensor to float
    input_tensor = input_tensor.float()

    # Get the number of dimensions of the input tensor
    num_dims = input_tensor.dim()

    # Prepare the structure element for convolution
    # Adding extra dimensions to match the input shape for convolution
    if num_dims == 2:  # For 2D inputs
        structure_element = structure_element.unsqueeze(0).unsqueeze(0).float()
    elif num_dims == 3:  # For 3D inputs, adding batch dimension
        structure_element = structure_element.unsqueeze(0).unsqueeze(0).float()
    else:
        raise ValueError("Input tensor must be 2D (X, Y) or 3D (X, Y, Z).")

    # Perform the convolution
    # if num_dims == 2:  # 2D convolution
    #     output = F.conv2d(input_tensor.unsqueeze(0).unsqueeze(0), structure_element, padding='same')
    # elif num_dims == 3:  # 3D convolution
    #     output = F.conv3d(input_tensor.unsqueeze(0).unsqueeze(0), structure_element, padding='same')
    output = torch.round(fft_conv(input_tensor.unsqueeze(0).unsqueeze(0), structure_element, padding='same'), decimals=0)

    # Threshold to get binary output
    output = output > 0

    # Squeeze the batch dimension out and convert to bool
    return output.squeeze(0).squeeze(0).bool()


def binary_erosion_torch(input_tensor, structure_element):
    return ~binary_dilation_torch(~input_tensor, structure_element)


def binary_opening_torch(input_tensor, structure_element):
    return binary_dilation_torch(binary_erosion_torch(input_tensor, structure_element), structure_element)


def binary_closing_torch(input_tensor, structure_element):
    return binary_erosion_torch(binary_dilation_torch(input_tensor, structure_element), structure_element)



class RandApplyRandomBinaryOperatorTransformd(MapTransform, RandomizableTransform):
    def __init__(self, keys,
                 channel_idx: Union[int, List[int], Tuple[int, ...]],
                 prob:float=0.5,
                 any_of_these: Tuple[Callable, ...] = (binary_dilation_torch, binary_erosion_torch, binary_closing_torch, binary_opening_torch),
                 strel_size: RandomScalar = (1, 10),
                 p_per_label: float = 1):
        
        super().__init__(keys)
        RandomizableTransform.__init__(self, prob)
        if not isinstance(channel_idx, (list, tuple)):
            channel_idx = [channel_idx]
        if isinstance(channel_idx, tuple):
            channel_idx = list(channel_idx)
        self.channel_idx = channel_idx
        self.any_of_these = any_of_these
        self.strel_size = strel_size
        self.p_per_label = p_per_label
    
    def get_parameters(self, data) -> dict:
        # this needs to be applied in random order to the channels
        np.random.shuffle(self.channel_idx)
        apply_to_channels = [self.channel_idx[i] for i, j in enumerate(torch.rand(len(self.channel_idx)) < self.p_per_label) if j]
        operators = [np.random.choice(self.any_of_these) for _ in apply_to_channels]
        strel_size = [sample_scalar(self.strel_size, image=data['label'], channel=a) for a in apply_to_channels]
        return {
            'apply_to_channels': apply_to_channels,
            'operators': operators,
            'strel_size': strel_size,
        }

    def __call__(self, data):
        d = dict(data)
        params = self.get_parameters(data)
        self.randomize(None)
        
        for key in self.keys:
            for a, o, s in zip(params['apply_to_channels'], params['operators'], params['strel_size']):
                if self._do_transform:
                    # this is a binary map so bool is fine
                    workon = d[key][a]#.numpy()
                    orig_dtype = workon.dtype
                    workon = workon.to(bool)
                    if workon.ndim == 2:
                        strel = disk(s, dtype=bool)
                    else:
                        strel = ball(s, dtype=bool)
                    result = o(workon, torch.from_numpy(strel))
                    other_ch = [i for i in self.channel_idx if i != a]
                    if len(other_ch) > 0:
                        was_added_mask = result & (~workon)
                        for oc in other_ch:
                            d[key][oc][was_added_mask] = 0
                    d[key][a] = result.to(orig_dtype)#torch.from_numpy(result)
        return d
    

class MirrorTransformd(MapTransform):
    def __init__(self, keys, allowed_axes: Tuple[int, ...]):
        super().__init__(keys)
        self.allowed_axes = allowed_axes
    
    def get_parameters(self) -> dict:
        axes = [i for i in self.allowed_axes if torch.rand(1) < 0.5]
        return {
            'axes': axes
        }
    
    def __call__(self, data):
        d = dict(data)
        params = self.get_parameters()
        
        if len(params['axes']) == 0:
            return d
        axes = [i + 1 for i in params['axes']]
        for key in self.keys:
            d[key] = torch.flip(d[key],axes)
        return d
    

class MaskImageTransformd(MapTransform):
    def __init__(self, keys, set_to:float = 0):
        super().__init__(keys)
        self.set_to = set_to
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key][d["label"]<0] = self.set_to
        return d


class RandRemoveRandomConnectedComponentFromOneHotEncodingTransformd(RandomizableTransform,MapTransform):
    def __init__(self, keys,
                 channel_idx: Union[int, List[int], Tuple[int, ...]],
                 prob:float = 0.5,
                 fill_with_other_class_p: float = 0.25,
                 dont_do_if_covers_more_than_x_percent: float = 0.25,
                 p_per_label: float = 1):
        super().__init__(keys)
        RandomizableTransform.__init__(self, prob)
        if not isinstance(channel_idx, (list, tuple)):
            channel_idx = [channel_idx]
        if isinstance(channel_idx, tuple):
            channel_idx = list(channel_idx)
        self.channel_idx = channel_idx
        self.fill_with_other_class_p = fill_with_other_class_p
        self.dont_do_if_covers_more_than_x_percent = dont_do_if_covers_more_than_x_percent
        self.p_per_label = p_per_label

    def get_parameters(self) -> dict:
        # this needs to be applied in random order to the channels
        np.random.shuffle(self.channel_idx)
        apply_to_channels = [self.channel_idx[i] for i, j in enumerate(torch.rand(len(self.channel_idx)) < self.p_per_label) if j]

        # self.fill_with_other_class_p cannot be resolved here because we don't know how many components there are
        return {
            'apply_to_channels': apply_to_channels,
        }
    
    def __call__(self, data):
        d = dict(data)
        params = self.get_parameters()
        if self._do_transform:
            for key in self.keys:
                for a in params['apply_to_channels']:
                    workon = d[key][a].to(bool).numpy()
                    if not np.any(workon):
                        continue
                    num_voxels = np.prod(workon.shape, dtype=np.uint64)
                    lab, component_sizes = label_with_component_sizes(workon.astype(bool))
                    if len(component_sizes) > 0:
                        valid_component_ids = [i for i, j in component_sizes.items() if j <
                                            num_voxels * self.dont_do_if_covers_more_than_x_percent]
                        # print('RemoveRandomConnectedComponentFromOneHotEncodingTransform', c,
                        # np.unique(data[b, c]), len(component_sizes), valid_component_ids,
                        # len(valid_component_ids))
                        if len(valid_component_ids) > 0:
                            random_component = np.random.choice(valid_component_ids)
                            d[key][a][lab == random_component] = 0
                            if np.random.uniform() < self.fill_with_other_class_p:
                                other_ch = [i for i in self.channel_idx if i != a]
                                if len(other_ch) > 0:
                                    other_class = np.random.choice(other_ch)
                                    d[key][other_class][lab == random_component] = 1
        return d

class RandConvertSegmentationToRegionsTransformd(RandomizableTransform,MapTransform):
    def __init__(self, keys,
                 regions: Union[List, Tuple], 
                 prob:float = 0.5,
                 channel_in_seg: int = 0):
        super().__init__(keys)
        RandomizableTransform.__init__(self, prob)
        self.regions = [torch.Tensor(i) if not isinstance(i, int) else torch.Tensor([i]) for i in regions]
        self.channel_in_seg = channel_in_seg
    
    def __call__(self, data):
        num_regions = len(self.regions)
        d = dict(data)
        if self._do_transform:
            for key in self.keys:
                region_output = torch.zeros((num_regions, d["label"].shape[1:]), dtype=torch.bool, device=d['label'].device)
                for region_id, region_labels in enumerate(self.regions):
                    if len(region_labels) == 1:
                        region_output[region_id] = d[key][self.channel_in_seg] == region_labels
                    else:
                        region_output[region_id] = torch.isin(d[key][self.channel_in_seg], region_labels)
                # we return bool here and leave it to the loss function to cast it to whatever it needs. Transferring bool to
                # device followed by cast on device should be faster than having fp32 here and transferring that
                d[key] = region_output
                return d
        else:
            return d

class DownsampleSegForDSTransformd(MapTransform):
    def __init__(self, keys,ds_scales: Union[List, Tuple]):
        super().__init__(keys)
        self.ds_scales = ds_scales
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            results = []
            for s in self.ds_scales:
                if not isinstance(s, (tuple, list)):
                    s = [s] * (d[key].ndim - 1)
                else:
                    assert len(s) == d[key].ndim - 1

                if all([i == 1 for i in s]):
                    results.append(d[key])
                else:
                    new_shape = [round(i * j) for i, j in zip(d[key].shape[1:], s)]
                    dtype = d[key].dtype
                    # interpolate is not defined for short etc
                    results.append(interpolate(d[key][None].float(), new_shape, mode='nearest-exact')[0].to(dtype))
            d[key] = results
        return d


class MoveSegAsOneHotToDataTransformd(MapTransform):
    def __init__(self, keys,
                  source_channel_idx: int, all_labels: Union[Tuple[int, ...], List[int]],
                 remove_channel_from_source: bool = True,
                 allow_missing_keys: bool = False):
        super().__init__(keys)
        
        """
        Used in nnU-Net to append segmentations from the previous stage to the image as additional input
        Args:
            source_channel_idx:
            all_labels:
            remove_channel_from_source:
        """
        self.source_channel_idx = source_channel_idx
        self.all_labels = all_labels
        self.remove_channel_from_source = remove_channel_from_source
    
    def __call__(self, d):
        seg = d['label'][self.source_channel_idx]
        
        seg_onehot = torch.zeros((len(self.all_labels), *seg.shape), dtype=d['image'].dtype)
        for i, l in enumerate(self.all_labels):
            seg_onehot[i][seg == l] = 1
        d['image'] = torch.cat((d['image'], seg_onehot))
        if self.remove_channel_from_source:
            remaining_channels = [i for i in range(d['label'].shape[0]) if i != self.source_channel_idx]
            d['label'] = d['label'][remaining_channels]
        return d




def generate_pos_label_else_rand_crop_centers(
    spatial_size: Sequence[int] | int,
    num_samples: int,
    pos_ratio: float,
    label_spatial_shape: Sequence[int],
    fg_indices: NdarrayOrTensor,
    bg_indices: NdarrayOrTensor,
    rand_state: np.random.RandomState | None = None,
    allow_smaller: bool = False,
) -> tuple[tuple]:
    """
    Generate valid sample locations based on the label with option for specifying foreground ratio
    Valid: samples sitting entirely within image, expected input shape: [C, H, W, D] or [C, H, W]

    Args:
        spatial_size: spatial size of the ROIs to be sampled.
        num_samples: total sample centers to be generated.
        pos_ratio: ratio of total locations generated that have center being foreground.
        label_spatial_shape: spatial shape of the original label data to unravel selected centers.
        fg_indices: pre-computed foreground indices in 1 dimension.
        bg_indices: pre-computed background indices in 1 dimension.
        rand_state: numpy randomState object to align with other modules.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).

    Raises:
        ValueError: When the proposed roi is larger than the image.
        ValueError: When the foreground and background indices lengths are 0.

    """
    if rand_state is None:
        rand_state = np.random.random.__self__  # type: ignore

    centers = []
    fg_indices = np.asarray(fg_indices) if isinstance(fg_indices, Sequence) else fg_indices
    bg_indices = np.asarray(bg_indices) if isinstance(bg_indices, Sequence) else bg_indices
    if len(fg_indices) == 0 and len(bg_indices) == 0:
        raise ValueError("No sampling location available.")

    if len(fg_indices) == 0 or len(bg_indices) == 0:
        pos_ratio = 0 if len(fg_indices) == 0 else 1
        warnings.warn(
            f"Num foregrounds {len(fg_indices)}, Num backgrounds {len(bg_indices)}, "
            f"unable to generate class balanced samples, setting `pos_ratio` to {pos_ratio}."
        )

    for _ in range(num_samples):
        if rand_state.rand() < pos_ratio:
            indices_to_use = fg_indices
        elif rand_state.rand() <= 0.5:
            indices_to_use = fg_indices
        else:
            indices_to_use = bg_indices
            
        random_int = rand_state.randint(len(indices_to_use))
        idx = indices_to_use[random_int]
        center = unravel_index(idx, label_spatial_shape).tolist()
        # shift center to range of valid centers
        centers.append(correct_crop_centers(center, spatial_size, label_spatial_shape, allow_smaller))

    return ensure_tuple(centers)



class RandCropByPosLabelForced(Randomizable, TraceableTransform, LazyTransform, MultiSampleTrait):
    """
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    And will return a list of arrays for all the cropped images.
    For example, crop two (3 x 3) arrays from (5 x 5) array with pos/neg=1::

        [[[0, 0, 0, 0, 0],
          [0, 1, 2, 1, 0],            [[0, 1, 2],     [[2, 1, 0],
          [0, 1, 3, 0, 0],     -->     [0, 1, 3],      [3, 0, 0],
          [0, 0, 0, 0, 0],             [0, 0, 0]]      [0, 0, 0]]
          [0, 0, 0, 0, 0]]]

    If a dimension of the expected spatial size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than expected size, and the cropped
    results of several images may not have exactly same shape.
    And if the crop ROI is partly out of the image, will automatically adjust the crop center to ensure the
    valid crop ROI.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `label` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        label: the label image that is used for finding foreground/background, if None, must set at
            `self.__call__`.  Non-zero indicates foreground, zero indicates background.
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image: optional image data to help select valid area, can be same as `img` or another image array.
            if not None, use ``label == 0 & image > image_threshold`` to select the negative
            sample (background) center. So the crop center will only come from the valid image areas.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to determine
            the valid image content areas.
        fg_indices: if provided pre-computed foreground indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.
        bg_indices: if provided pre-computed background indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    backend = SpatialCrop.backend

    def __init__(
        self,
        spatial_size: Sequence[int] | int,
        label: torch.Tensor | None = None,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image: torch.Tensor | None = None,
        image_threshold: float = 0.0,
        fg_indices: NdarrayOrTensor | None = None,
        bg_indices: NdarrayOrTensor | None = None,
        allow_smaller: bool = False,
        lazy: bool = False,
    ) -> None:
        LazyTransform.__init__(self, lazy)
        self.spatial_size = spatial_size
        self.label = label
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image = image
        self.image_threshold = image_threshold
        self.centers: tuple[tuple] | None = None
        self.fg_indices = fg_indices
        self.bg_indices = bg_indices
        self.allow_smaller = allow_smaller

    def randomize(
        self,
        label: torch.Tensor | None = None,
        fg_indices: NdarrayOrTensor | None = None,
        bg_indices: NdarrayOrTensor | None = None,
        image: torch.Tensor | None = None,
    ) -> None:
        fg_indices_ = self.fg_indices if fg_indices is None else fg_indices
        bg_indices_ = self.bg_indices if bg_indices is None else bg_indices
        if fg_indices_ is None or bg_indices_ is None:
            if label is None:
                raise ValueError("label must be provided.")
            fg_indices_, bg_indices_ = map_binary_to_indices(label, image, self.image_threshold)
        _shape = None
        if label is not None:
            _shape = label.peek_pending_shape() if isinstance(label, MetaTensor) else label.shape[1:]
        elif image is not None:
            _shape = image.peek_pending_shape() if isinstance(image, MetaTensor) else image.shape[1:]
        if _shape is None:
            raise ValueError("label or image must be provided to get the spatial shape.")
        self.centers = generate_pos_label_else_rand_crop_centers(
            self.spatial_size,
            self.num_samples,
            self.pos_ratio,
            _shape,
            fg_indices_,
            bg_indices_,
            self.R,
            self.allow_smaller,
        )

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, _val: bool):
        self._lazy = _val

    @property
    def requires_current_data(self):
        return False

    def __call__(
        self,
        img: torch.Tensor,
        label: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        fg_indices: NdarrayOrTensor | None = None,
        bg_indices: NdarrayOrTensor | None = None,
        randomize: bool = True,
        lazy: bool | None = None,
    ) -> list[torch.Tensor]:
        """
        Args:
            img: input data to crop samples from based on the pos/neg ratio of `label` and `image`.
                Assumes `img` is a channel-first array.
            label: the label image that is used for finding foreground/background, if None, use `self.label`.
            image: optional image data to help select valid area, can be same as `img` or another image array.
                use ``label == 0 & image > image_threshold`` to select the negative sample(background) center.
                so the crop center will only exist on valid image area. if None, use `self.image`.
            fg_indices: foreground indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.
            bg_indices: background indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.
            randomize: whether to execute the random operations, default to `True`.
            lazy: a flag to override the lazy behaviour for this call, if set. Defaults to None.

        """
        if image is None:
            image = self.image
        if randomize:
            if label is None:
                label = self.label
            self.randomize(label, fg_indices, bg_indices, image)
        results: list[torch.Tensor] = []
        if self.centers is not None:
            img_shape = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
            roi_size = fall_back_tuple(self.spatial_size, default=img_shape)
            lazy_ = self.lazy if lazy is None else lazy
            for i, center in enumerate(self.centers):
                cropper = SpatialCrop(roi_center=center, roi_size=roi_size, lazy=lazy_)
                cropped = cropper(img)
                if get_track_meta():
                    ret_: MetaTensor = cropped  # type: ignore
                    ret_.meta[Key.PATCH_INDEX] = i
                    ret_.meta["crop_center"] = center
                    self.push_transform(ret_, replace=True, lazy=lazy_)
                results.append(cropped)
        return results



class RandCropByPosLabelForcedd(Randomizable, MapTransform, LazyTransform, MultiSampleTrait):
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropByPosNegLabel`.
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    Suppose all the expected fields specified by `keys` have same shape,
    and add `patch_index` to the corresponding metadata.
    And will return a list of dictionaries for all the cropped images.

    If a dimension of the expected spatial size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected size,
    and the cropped results of several images may not have exactly the same shape.
    And if the crop ROI is partly out of the image, will automatically adjust the crop center
    to ensure the valid crop ROI.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding foreground/background.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `data[label_key]` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image_key: if image_key is not None, use ``label == 0 & image > image_threshold`` to select
            the negative sample(background) center. so the crop center will only exist on valid image area.
        image_threshold: if enabled image_key, use ``image > image_threshold`` to determine
            the valid image content area.
        fg_indices_key: if provided pre-computed foreground indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        bg_indices_key: if provided pre-computed background indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    backend = RandCropByPosLabelForced.backend

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Sequence[int] | int,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image_key: str | None = None,
        image_threshold: float = 0.0,
        fg_indices_key: str | None = None,
        bg_indices_key: str | None = None,
        allow_smaller: bool = False,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy)
        self.label_key = label_key
        self.image_key = image_key
        self.fg_indices_key = fg_indices_key
        self.bg_indices_key = bg_indices_key
        self.cropper = RandCropByPosLabelForced(
            spatial_size=spatial_size,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
            image_threshold=image_threshold,
            allow_smaller=allow_smaller,
            lazy=lazy,
        )

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ):
        super().set_random_state(seed, state)
        self.cropper.set_random_state(seed, state)
        return self

    def randomize(
        self,
        label: torch.Tensor | None = None,
        fg_indices: NdarrayOrTensor | None = None,
        bg_indices: NdarrayOrTensor | None = None,
        image: torch.Tensor | None = None,
    ) -> None:
        self.cropper.randomize(label=label, fg_indices=fg_indices, bg_indices=bg_indices, image=image)

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, value: bool) -> None:
        self._lazy = value
        self.cropper.lazy = value

    @property
    def requires_current_data(self):
        return True

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None
    ) -> list[dict[Hashable, torch.Tensor]]:
        d = dict(data)
        fg_indices = d.pop(self.fg_indices_key, None)
        bg_indices = d.pop(self.bg_indices_key, None)

        self.randomize(d.get(self.label_key), fg_indices, bg_indices, d.get(self.image_key))

        # initialize returned list with shallow copy to preserve key ordering
        ret: list = [dict(d) for _ in range(self.cropper.num_samples)]
        # deep copy all the unmodified data
        for i in range(self.cropper.num_samples):
            for key in set(d.keys()).difference(set(self.keys)):
                ret[i][key] = deepcopy(d[key])

        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(d):
            for i, im in enumerate(self.cropper(d[key], randomize=False, lazy=lazy_)):
                ret[i][key] = im
        return ret




class RandMultiplicativeBrightnessTransformdd(MapTransform,RandomizableTransform):
    def __init__(self, keys, multiplier_range, synchronize_channels: bool, p_per_channel: float = 1,prob:float = 0.5) -> None:
        super().__init__(keys)
        RandomizableTransform.__init__(self, prob)
        self.multiplier_range = multiplier_range
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = p_per_channel
    
    def get_parameters(self, data) -> dict:
        shape = data['image'].shape
        apply_to_channel = torch.where(torch.rand(shape[0]) < self.p_per_channel)[0]
        if self.synchronize_channels:
            multipliers = torch.Tensor([sample_scalar(self.multiplier_range, image=data['image'], channel=None)] * len(apply_to_channel))
        else:
            multipliers = torch.Tensor([sample_scalar(self.multiplier_range, image=data['image'], channel=c) for c in apply_to_channel])
        return {
            'apply_to_channel': apply_to_channel,
            'multipliers': multipliers
        }
        
    def __call__(self,data):
        d = dict(data)
        params = self.get_parameters(data)
        for key in self.keys:
            if len(params['apply_to_channel']) == 0:
                continue
            # even though this is array notation it's a lot slower. Shame shame
            # img[params['apply_to_channel']] *= params['multipliers'].view(-1, *[1]*(img.ndim - 1))
            for c, m in zip(params['apply_to_channel'], params['multipliers']):
                d[key][c] *= m
        return d










###preprocess

class TransposeCropnonzerod(MapTransform):
    def __init__(self, keys, transpose_forward:list) -> None:
        super().__init__(keys)
        self.transpose_forward = [i + 1 for i in transpose_forward]
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = d[key].permute(0,*[self.transpose_forward[2],self.transpose_forward[1],self.transpose_forward[0]])
        
        
        d["image"], d["label"], _ = crop_to_nonzero(d["image"], d["label"])
        return d


    

class CTNormalizationd(MapTransform):
    def __init__(self, keys, percentile_00_5:float,percentile_99_5:float,mean:float,std:float) -> None:
        super().__init__(keys)
        self.percentile_00_5 = percentile_00_5
        self.percentile_99_5 = percentile_99_5
        self.mean = mean
        self.std = std
        
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            image = torch.clamp(image, min=self.percentile_00_5, max=self.percentile_99_5)
            image -= self.mean
            image /= max(self.std, 1e-8)
            d[key] = image
        return d


class nnUNetResampling(MapTransform):
    def __init__(self, keys,mode,target_spacing) -> None:
        super().__init__(keys)
        self.mode = mode
        self.target_spacing = target_spacing
    
    def __call__(self, data):
        d = dict(data)
        for idx, key in enumerate(self.keys):
            original_spacing = d[key].meta["spacing"]
            original_spacing = [original_spacing[2],original_spacing[1],original_spacing[0]]
            
            new_shape = compute_new_shape(d[key].shape[1:], original_spacing, self.target_spacing)
            resizer = Resize(spatial_size=new_shape,mode=self.mode[idx])
            d[key] = resizer(d[key])
        
        return d