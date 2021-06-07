"""
Reference: https://github.com/okankop/vidaug
"""

import numpy as np
import PIL
import random
import math


class TimeReversal(object):
    """
    Time Reverse the video 
    """
    
    def __call__(self, clip):
       #out = clip[:self.size]
       for i in range(len(clip)):
           out.append(clip[len(clip)-1-i])
       return out