""" 
    Define a set of transformations which can be applied simultaneously to the input image and its corresponding labels.

    This is relevant for the task of semantic segmentation since the input image and its label need to be treated in the same way.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from PIL.Image import Image as PILImage
import torch
import torchvision
import torchvision.transforms.functional as TF


class JointTransformation(ABC):
    """General transformation which can be applied simultaneously to the raw image, input image and its corresponding anntations."""

    @abstractmethod
    def __call__(
        self, image: torch.Tensor, label: torch.Tensor, raw_image: Optional[PILImage] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[PILImage]]:
        """Apply a transformation to a given image and its corresponding label.

        Args:
          image (torch.Tensor): input image to be transformed.
          label (torch.Tensor): label to be transformed.
          raw_image (Optional[PILImage], optional): optional because
          it is useful to transform for visualization purposes.

        Returns:
          Tuple[torch.Tensor, torch.Tensor]: transformed image and its corresponding label
        """
        raise NotImplementedError


def assert_dimensions_are_equal(image: torch.Tensor, label: torch.Tensor, raw_image: Optional[PILImage] = None) -> bool:
    """Check if the dimensions of the input image and its corresponding label are identical.

    Args:
        image (torch.Tensor): input image
        label (torch.Tensor): label
        raw_image (Optional[PILImage], optional): optional because
          it is useful to transform for visualization purposes.

    Returns:
        bool: True if dimensions are identical, False otherwise.
    """
    if raw_image is not None:
        assert image.shape[1] == label.shape[1] == raw_image.height, "Dimensions of all input should be identical."
        assert image.shape[2] == label.shape[2] == raw_image.width, "Dimensions of all input should be identical."
    else:
        assert image.shape[1] == label.shape[1], "Dimensions of all input should be identical."
        assert image.shape[2] == label.shape[2], "Dimensions of all input should be identical."


class JointCenterCropTransform(JointTransformation):
    """Extract a patch from the image center."""

    def __init__(self, crop_height: Optional[int] = None, crop_width: Optional[int] = None):
        """Set height and width of cropping region.

        Args:
            crop_height (Optional[int], optional): Height of cropping region. Defaults to None.
            crop_width (Optional[int], optional): Width of cropping region. Defaults to None.
        """
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor, raw_image: Optional[PILImage] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[PILImage]]:
        # dimension of each input should be identical
        assert_dimensions_are_equal(image, label, raw_image)

        if (self.crop_height is None) or (self.crop_width is None):
            if raw_image is not None:
                return image, label, raw_image
            else:
                return image, label

        img_chans, img_height, img_width = image.shape[:3]
        label_chans = label.shape[0]

        if self.crop_width > img_width:
            raise ValueError("Width of cropping region must not be greather than img width")
        if self.crop_height > img_height:
            raise ValueError("Height of cropping region must not be greather than img height.")

        image_cropped = TF.center_crop(image, [self.crop_height, self.crop_width])
        label_cropped = TF.center_crop(label, [self.crop_height, self.crop_width])
        assert image_cropped.shape[0] == img_chans, "Cropped image has an unexpected number of channels."
        assert image_cropped.shape[1] == self.crop_height, "Cropped image has not the desired size."
        assert image_cropped.shape[2] == self.crop_width, "Cropped image has not the desired width."

        assert label_cropped.shape[0] == label_chans, "Cropped label has an unexpected number of channels."
        assert label_cropped.shape[1] == self.crop_height, "Cropped label has not the desired size."
        assert label_cropped.shape[2] == self.crop_width, "Cropped label has not the desired width."

        if raw_image is not None:
            raw_image_cropped = TF.center_crop(raw_image, [self.crop_height, self.crop_width])
            assert raw_image_cropped.height == self.crop_height, "Cropped raw image has not the desired size."
            assert raw_image_cropped.width == self.crop_width, "Cropped raw image has not the desired width."
            return image_cropped, label_cropped, raw_image_cropped

        else:
            return image_cropped, label_cropped


class JointRandomCropTransform(JointTransformation):
    """Extract a random patch from a given image and its corresponding label."""

    def __init__(self, crop_coef: Optional[float] = None):
        """Set height and width of cropping region.

        Args:
            crop_height (Optional[int], optional): Height of cropping region. Defaults to None.
            crop_width (Optional[int], optional): Width of cropping region. Defaults to None.
        """
        self.crop_coef = crop_coef

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor, raw_image: Optional[PILImage] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[PILImage]]:
        """Apply cropping to an image and its corresponding label.

        Args:
            image (torch.Tensor): image to be cropped of shape [C x H x W]
            label (torch.Tensor): label to be cropped of shape [1 x H x W]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: cropped image, cropped label
        """
        # dimension of each input should be identical
        assert_dimensions_are_equal(image, label, raw_image)

        img_chans, img_height, img_width = image.shape[:3]
        self.crop_width = int(img_width * self.crop_coef)
        self.crop_height = int(img_height * self.crop_coef)
        label_chans = label.shape[0]

        if self.crop_width > img_width:
            raise ValueError("Width of cropping region must not be greather than img width")
        if self.crop_height > img_height:
            raise ValueError("Height of cropping region must not be greather than img height.")

        max_x = img_width - self.crop_width
        x_start = random.randint(0, max_x)

        max_y = img_height - self.crop_height
        y_start = random.randint(0, max_y)

        assert (x_start + self.crop_width) <= img_width, "Cropping region (width) exceeds image dims."
        assert (y_start + self.crop_height) <= img_height, "Cropping region (height) exceeds image dims."

        #image_cropped = TF.crop(image, y_start, x_start, self.crop_height, self.crop_width)
        image_cropped = image[:, y_start : y_start + self.crop_height, x_start : x_start + self.crop_width]
        #label_cropped = TF.crop(label, y_start, x_start, self.crop_height, self.crop_width)
        label_cropped = label[:, y_start : y_start + self.crop_height, x_start : x_start + self.crop_width]

        assert image_cropped.shape[0] == img_chans, "Cropped image has an unexpected number of channels."
        assert image_cropped.shape[1] == self.crop_height, "Cropped image has not the desired size."
        assert image_cropped.shape[2] == self.crop_width, "Cropped image has not the desired width."

        assert label_cropped.shape[0] == label_chans, "Cropped label has an unexpected number of channels."
        assert label_cropped.shape[1] == self.crop_height, "Cropped label has not the desired size."
        assert label_cropped.shape[2] == self.crop_width, "Cropped label has not the desired width."

        if raw_image is not None:
            #raw_image_cropped = TF.crop(raw_image, y_start, x_start, self.crop_height, self.crop_width)
            crop_box = (x_start, y_start, x_start + self.crop_width, y_start + self.crop_height)
            raw_image_cropped = raw_image.crop(crop_box)
            assert raw_image_cropped.height == self.crop_height, "Cropped raw image has not the desired size."
            assert raw_image_cropped.width == self.crop_width, "Cropped raw image has not the desired width."
            return image_cropped, label_cropped, raw_image_cropped
        else:
            return image_cropped, label_cropped


class JointResizeTransform(JointTransformation):
    """Resize a given image and its corresponding label."""

    def __init__(self, height: Optional[int] = None, width: Optional[int] = None):
        """Set params for resize operation.

        Args:
            width (Optional[int], optional): New width dimension. Defaults to None.
            height (Optional[int], optional): New height dimension. Defaults to None.
            keep_aspect_ratio (bool, optional): Specify if aspect ratio should stay the same. Defaults to False.
        """
        if width is not None:
            if not isinstance(width, int):
                raise ValueError("width must be of type int")
        self.resized_width = width

        if height is not None:
            if not isinstance(height, int):
                raise ValueError("height must be of type int")
        self.resized_height = height


    def __call__(
        self, image: torch.Tensor, label: torch.Tensor, raw_image: Optional[PILImage] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[PILImage]]:
        """Apply resizing to an image and its corresponding label.

        Args:
            image (torch.Tensor): image to be resized of shape [C x H x W]
            label (torch.Tensor): label to be cropped of shape [C x H x W]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: [description]
        """
        # dimension of each input should be identical.
        # assert image.shape[1] == label.shape[1], "Dimensions of all input should be identical."
        # assert image.shape[2] == label.shape[2], "Dimensions of all input should be identical."

        if (self.resized_width is None) and (self.resized_height is None):
            if raw_image is not None:
                return image, label, raw_image
            else:
                return image, label

        # original image dimension
        h, w = image.shape[1], image.shape[2]

        # We provide width and height
        image_resized = TF.resize(image, [self.resized_height, self.resized_width], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)  # type: ignore
        label_resized = TF.resize(label, [self.resized_height, self.resized_width], interpolation=TF.InterpolationMode.NEAREST, antialias=True)  # type: ignore

        if raw_image is not None:
            raw_image_resized = TF.resize(
                raw_image,
                [self.resized_height, self.resized_width],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            return image_resized, label_resized, raw_image_resized
        else:
            return image_resized, label_resized


class JointRandomRotationTransform(JointTransformation):
    """Rotate a given image and its corresponding label."""

    def __init__(
        self, min_angle: Optional[float] = None, max_angle: Optional[float] = None, step_size: Optional[float] = None
    ):
        self.min_angle = min_angle  # degree
        self.max_angle = max_angle  # degree
        self.step_size = step_size  # degree

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor, raw_image: Optional[PILImage] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[PILImage]]:
        if (self.min_angle is None) or (self.max_angle is None) or (self.step_size is None):
            return image, label

        assert self.min_angle is not None
        assert self.max_angle is not None
        assert self.step_size is not None

        assert self.min_angle < self.max_angle
        assert self.step_size > 0

        angles = torch.arange(self.min_angle, self.max_angle, step=self.step_size)
        random_angle = float(random.choice(list(angles)))  # degree

        image_rotated = TF.rotate(image, random_angle, interpolation=TF.InterpolationMode.BILINEAR)
        label_rotated = TF.rotate(label, random_angle, interpolation=TF.InterpolationMode.NEAREST)

        if raw_image is not None:
            raw_image_rotated = TF.rotate(raw_image, random_angle, interpolation=TF.InterpolationMode.BILINEAR)
            return image_rotated, label_rotated, raw_image_rotated
        else:
            return image_rotated, label_rotated


class JointRandomColorJitterTransform(JointTransformation):
    """Apply colour jitter to a given image.
    Ignores the label."""

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ):
        """Set colour jitter parameters.
        See: https://pytorch.org/vision/stable/transforms.html
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor, raw_image: Optional[PILImage] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        jitter = torchvision.transforms.ColorJitter(
            brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue
        )
        image_jitted = jitter(image)
        if raw_image is not None:
            raw_image_jitted = jitter(raw_image)
            return image_jitted, label, raw_image_jitted
        else:
            return image_jitted, label

class JointRandomFlip(JointTransformation):
    """Apply horizontal flip to an image"""

    def __init__(
        self,
        prob_th: float = 0.5,
    ):
        self.prob_th = prob_th

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor, raw_image: Optional[PILImage] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        #Generate a random number between 0 and 1
        coin = random.random()
        if coin > self.prob_th:
            image = TF.hflip(image)
            label = TF.hflip(label)
            if raw_image is not None:
                raw_image = TF.hflip(raw_image)

        if raw_image is not None:
            return image, label, raw_image
        else:
            return image, label


# TODO: Move this somewhere else? Config file or here?
def get_transformations(cfg, stage: str) -> List[JointTransformation]:
    assert stage in ["train", "val", "test"]
    transformations = []

    try:
        if cfg[stage]["transformations"] is None:
            return transformations

        for tf_name in cfg[stage]["transformations"].keys():  
            if tf_name == "random_flip":
                print('RANDOM FPLIP TRANSFORMATION')
                prob_th = cfg[stage]["transformations"][tf_name]["prob"]
                transformer = JointRandomFlip(prob_th)
                transformations.append(transformer)
                
            if tf_name == "random_crop":
                print('CROPPING TRANSFORMATION')
                crop_coef = cfg[stage]["transformations"][tf_name]["coeff"]
                transformer = JointRandomCropTransform(crop_coef)
                transformations.append(transformer)
                
            if tf_name == "resize":
                print('RESIZE TRANSFORMATION')
                resize_height = cfg[stage]["transformations"][tf_name]["height"]
                resize_width = cfg[stage]["transformations"][tf_name]["width"]
                transformer = JointResizeTransform(resize_height, resize_width)
                transformations.append(transformer)
            
            """
            if tf_name == "random_rotation":
                min_angle = cfg[stage]["transformations"][tf_name]["min_angle"]
                max_angle = cfg[stage]["transformations"][tf_name]["max_angle"]
                step_size = cfg[stage]["transformations"][tf_name]["step_size"]
                transformer = JointRandomRotationTransform(min_angle, max_angle, step_size)
                transformations.append(transformer)

            if tf_name == "center_crop":
                crop_height = cfg[stage]["transformations"][tf_name]["height"]
                crop_width = cfg[stage]["transformations"][tf_name]["width"]
                transformer = JointCenterCropTransform(crop_height, crop_width)
                transformations.append(transformer)

            if tf_name == "color_jitter":
                brightness = cfg[stage]["transformations"][tf_name]["brightness"]
                contrast = cfg[stage]["transformations"][tf_name]["contrast"]
                saturation = cfg[stage]["transformations"][tf_name]["saturation"]
                hue = cfg[stage]["transformations"][tf_name]["hue"]
                transformer = JointRandomColorJitterTransform(brightness, contrast, saturation, hue)
                transformations.append(transformer) 
            """
            

    except KeyError:
        return transformations

    return transformations
