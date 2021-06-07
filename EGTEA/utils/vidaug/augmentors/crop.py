"""
Augmenters that apply video flipping horizontally and
vertically.

To use the augmenters, clone the complete repo and use
`from vidaug import augmenters as va`
and then e.g. :
    seq = va.Sequential([ va.HorizontalFlip(),
                          va.VerticalFlip() ])

List of augmenters:
    * CenterCrop
    * CornerCrop
    * RandomCrop
    
Reference: https://github.com/okankop/vidaug
"""

import numpy as np
import PIL
import numbers
import random
import cv2 as cv


class CenterCrop(object):
    """
    Extract center crop of thevideo.

    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size

    def __call__(self, clip):
        crop_h, crop_w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        if crop_w > im_w or crop_h > im_h:
            error_msg = ('Initial image size should be larger then' +
                         'cropped size but got cropped sizes : ' +
                         '({w}, {h}) while initial image is ({im_w}, ' +
                         '{im_h})'.format(im_w=im_w, im_h=im_h, w=crop_w,
                                          h=crop_h))
            raise ValueError(error_msg)

        w1 = int(round((im_w - crop_w) / 2.))
        h1 = int(round((im_h - crop_h) / 2.))

        if isinstance(clip[0], np.ndarray):
            return [img[h1:h1 + crop_h, w1:w1 + crop_w, :] for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.crop((w1, h1, w1 + crop_w, h1 + crop_h)) for img in clip]


class CornerCrop(object):
    """
    Extract corner crop of the video.

    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).

        crop_position (str): Selected corner (or center) position from the
        list ['c', 'tl', 'tr', 'bl', 'br']. If it is non, crop position is
        selected randomly at each call.
    """

    def __init__(self, size, crop_position=None):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size

        if crop_position is None:
            self.randomize = True
        else:
            if crop_position not in ['c', 'tl', 'tr', 'bl', 'br']:
                raise ValueError("crop_position should be one of " +
                                 "['c', 'tl', 'tr', 'bl', 'br']")
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, clip):
        crop_h, crop_w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(0,len(self.crop_positions) - 1)]

        if self.crop_position == 'c':
            th, tw = (self.size, self.size)
            x1 = int(round((im_w - crop_w) / 2.))
            y1 = int(round((im_h - crop_h) / 2.))
            x2 = x1 + crop_w
            y2 = y1 + crop_h
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_w
            y2 = crop_h
        elif self.crop_position == 'tr':
            x1 = im_w - crop_w
            y1 = 0
            x2 = im_w
            y2 = crop_h
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = im_h - crop_h
            x2 = crop_w
            y2 = im_h
        elif self.crop_position == 'br':
            x1 = im_w - crop_w
            y1 = im_h - crop_h
            x2 = im_w
            y2 = im_h

        if isinstance(clip[0], np.ndarray):
            return [img[y1:y2, x1:x2, :] for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.crop((x1, y1, x2, y2)) for img in clip]


class RandomCrop(object):
    """
    Extract random crop of the video.

    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).

        crop_position (str): Selected corner (or center) position from the
        list ['c', 'tl', 'tr', 'bl', 'br']. If it is non, crop position is
        selected randomly at each call.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size

    def __call__(self, clip):
        crop_h, crop_w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        if crop_w > im_w or crop_h > im_h:
            error_msg = ('Initial image size should be larger then' +
                         'cropped size but got cropped sizes : ' +
                         '({w}, {h}) while initial image is ({im_w}, ' +
                         '{im_h})'.format(im_w=im_w, im_h=im_h, w=crop_w,
                                          h=crop_h))
            raise ValueError(error_msg)

        w1 = random.randint(0, im_w - crop_w)
        h1 = random.randint(0, im_h - crop_h)

        if isinstance(clip[0], np.ndarray):
            return [img[h1:h1 + crop_h, w1:w1 + crop_w, :] for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.crop((w1, h1, w1 + crop_w, h1 + crop_h)) for img in clip]

class MultiCrop(object):
    """
    Extract corner crop of the video.

    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).

        crop_position (str): Selected corner (or center) position from the
        list ['c', 'tl', 'tr', 'bl', 'br']. If it is non, crop position is
        selected randomly at each call.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size

        #if crop_position is None:
        #    self.randomize = True
        #else:
        #    if crop_position not in ['c', 'tl', 'tr', 'bl', 'br']:
        #        raise ValueError("crop_position should be one of " +
        #                         "['c', 'tl', 'tr', 'bl', 'br']")
        #    self.randomize = False
        #self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, clip):
        crop_h, crop_w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        images = []
        for crop_position in self.crop_positions:
          if crop_position == 'c':
             th, tw = (self.size, self.size)
             x1 = int(round((im_w - crop_w) / 2.))
             y1 = int(round((im_h - crop_h) / 2.))
             x2 = x1 + crop_w
             y2 = y1 + crop_h
          elif crop_position == 'tl':
             x1 = 0
             y1 = 0
             x2 = crop_w
             y2 = crop_h
          elif crop_position == 'tr':
             x1 = im_w - crop_w
             y1 = 0
             x2 = im_w
             y2 = crop_h
          elif crop_position == 'bl':
             x1 = 0
             y1 = im_h - crop_h
             x2 = crop_w
             y2 = im_h
          elif crop_position == 'br':
             x1 = im_w - crop_w
             y1 = im_h - crop_h
             x2 = im_w
             y2 = im_h
          if isinstance(clip[0], np.ndarray):
             images.extend([img[y1:y2, x1:x2, :] for img in clip])
          elif isinstance(clip[0], PIL.Image.Image):
             images.extend([img.crop((x1, y1, x2, y2)) for img in clip])
        return images

#class MultiScaleCrop(object):
#   def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
#        self.scales = scales if scales is not None else [1, .875, .75, .66]
#        self.max_distort = max_distort
#        self.fix_crop = fix_crop
#        self.more_fix_crop = more_fix_crop
#        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
#        self.interpolation = cv.INTER_LINEAR
#   def __call__(self, img_group):
#        im_size = img_group[0].shape
#        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
#        crop_img_group = [img[offset_w:offset_w + crop_w, offset_h:offset_h + crop_h] for img in img_group]
#        ret_img_group = [cv.resize(img, self.input_size, self.interpolation) for img in ret_img_group]
#        return ret_img_group
   
#   def _sample_crop_size(self, im_size):
#        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
#        base_size = min(image_w, image_h)
#        crop_sizes = [int(base_size * x) for x in self.scales]
#        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
#        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

#        pairs = []
#        for i, h in enumerate(crop_h):
#            for j, w in enumerate(crop_w):
#                if abs(i - j) <= self.max_distort:
#                    pairs.append((w, h))

#        crop_pair = random.choice(pairs)
#        if not self.fix_crop:
#            w_offset = random.randint(0, image_w - crop_pair[0])
#            h_offset = random.randint(0, image_h - crop_pair[1])
#        else:
#            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

#        return crop_pair[0], crop_pair[1], w_offset, h_offset

#   def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
#        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
#        return random.choice(offsets)

#   @staticmethod
#   def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
#        w_step = (image_w - crop_w) / 4
#        h_step = (image_h - crop_h) / 4

#        ret = list()
#        ret.append((0, 0))  # upper left
#        ret.append((4 * w_step, 0))  # upper right
#        ret.append((0, 4 * h_step))  # lower left
#        ret.append((4 * w_step, 4 * h_step))  # lower right
#        ret.append((2 * w_step, 2 * h_step))  # center#

#        if more_fix_crop:
#           ret.append((0, 2 * h_step))  # center left
#            ret.append((4 * w_step, 2 * h_step))  # center right
#            ret.append((2 * w_step, 4 * h_step))  # lower center
#            ret.append((2 * w_step, 0 * h_step))  # upper center#

#            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
#            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
#            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
#            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

#        return ret

class MultiRandomCrop(object):
    """
    Extract random crop of the video.

    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).

        crop_position (str): Selected corner (or center) position from the
        list ['c', 'tl', 'tr', 'bl', 'br']. If it is non, crop position is
        selected randomly at each call.
    """

    def __init__(self, how_many, size):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size
        self.how_many = how_many

    def __call__(self, clip):
        new_clip = []
        for i in range(self.how_many):
            crop_h, crop_w = self.size
            if isinstance(clip[0], np.ndarray):
                im_h, im_w, im_c = clip[0].shape
            elif isinstance(clip[0], PIL.Image.Image):
                im_w, im_h = clip[0].size
            else:
              raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

            if crop_w > im_w or crop_h > im_h:
                error_msg = ('Initial image size should be larger then' +
                         'cropped size but got cropped sizes : ' +
                         '({w}, {h}) while initial image is ({im_w}, ' +
                         '{im_h})'.format(im_w=im_w, im_h=im_h, w=crop_w,
                                          h=crop_h))
                raise ValueError(error_msg)

            w1 = random.randint(0, im_w - crop_w)
            h1 = random.randint(0, im_h - crop_h)
            if isinstance(clip[0], np.ndarray):
              #new_clip.append([img[h1:h1 + crop_h, w1:w1 + crop_w, :] for img in clip])
              for img in clip:
                   new_clip.append(img[h1:h1 + crop_h, w1:w1 + crop_w, :])
            elif isinstance(clip[0], PIL.Image.Image):
              new_clip.append([img.crop((w1, h1, w1 + crop_w, h1 + crop_h)) for img in clip])
        #print(len(new_clip))
        return new_clip 
