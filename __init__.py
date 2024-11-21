"""
@author: dfl
@title: CLIP with BREAK syntax
@nickname: CLIP with BREAK
@description: CLIP text encoder that does BREAK prompting like A1111
"""

from .nodes import *

NODE_CLASS_MAPPINGS = {
    "ImageToAscii": ImageToAscii,          
}
