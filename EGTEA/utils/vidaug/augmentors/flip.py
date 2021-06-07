"""
Augmenters that apply video flipping horizontally and
vertically.

To use the augmenters, clone the complete repo and use
`from vidaug import augmenters as va`
and then e.g. :
    seq = va.Sequential([ va.HorizontalFlip(),
                          va.VerticalFlip() ])

List of augmenters:
    * HorizontalFlip
    * VerticalFlip

Reference: https://github.com/okankop/vidaug
"""

import numpy as np
import PIL


class HorizontalFlip(object):
    """
    Horizontally flip the video.
    """

    def __call__(self, clip):
        if isinstance(clip[0], np.ndarray):
            return [np.fliplr(img) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            ' but got list of {0}'.format(type(clip[0])))



class VerticalFlip(object):
    """
    Vertically flip the video.
    """

    def __call__(self, clip):
        if isinstance(clip[0], np.ndarray):
            return [np.flipud(img) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.transpose(PIL.Image.FLIP_TOP_BOTTOM) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            ' but got list of {0}'.format(type(clip[0])))



class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [np.fliplr(img) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = np.invert(ret[i])  # invert flow pixel values when flipping
            return ret
        else:
            return img_group
