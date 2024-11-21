from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import Tensor
from xlib.file import SplittedFile
from xlib.image import ImageProcessor
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)

import tyro
from .repo.src.config.argument_config import ArgumentConfig
from .repo.src.config.inference_config import InferenceConfig
from .repo.src.config.crop_config import CropConfig
from .repo.src.live_portrait_wrapper import LivePortraitWrapperAnimal
from .repo.src.utils.camera import get_rotation_matrix
from .repo.src.utils.helper import calc_motion_multiplier

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

class LivePortrait:
    """
    Latent Image Animator: Learning to Animate Images via Latent Space Navigation
    https://github.com/wyhsirius/LIA

    arguments

     device_info    ORTDeviceInfo

        use LIA.get_available_devices()
        to determine a list of avaliable devices accepted by model

    raises
     Exception
    """

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info : ORTDeviceInfo):
        if device_info not in LivePortrait.get_available_devices():
            raise Exception(f'device_info {device_info} is not in available devices for LIA')
        # generator_path = Path(__file__).parent / 'generator.onnx'
        # SplittedFile.merge(generator_path, delete_parts=False)
        # if not generator_path.exists():
        #     raise FileNotFoundError(f'{generator_path} not found')
            
        # self._generator = InferenceSession_with_device(str(generator_path), device_info)
        
        inference_config = InferenceConfig()
        self.live_portrait_wrapper_animal = LivePortraitWrapperAnimal(inference_config)
        
        self.kp_info_s = None
        self.f_s = None
        self.x_s = None
        
        self.x_d_0 = None
        self.motion_multiplier = None


    def get_input_size(self):
        """
        returns optimal (Width,Height) for input images, thus you can resize source image to avoid extra load
        """
        return (256,256)
    
    def clear_source_cache(self):
        self.kp_info_s = None

    def clear_ref_motion_cache(self):
        self.x_d_0 = None

    def cap_value(self, value:Tensor, cap:float, max:float, threshold:float = 0.8):
        if (torch.abs(value) < cap * threshold):
            return value
        elif torch.abs(value) > max:
            return torch.clamp(value, -cap, cap)
        else:
            if value > 0:
                return (cap - threshold * cap) / (max - threshold * cap) * (value - threshold * cap) + threshold * cap
            else:
                return (-threshold * cap - (-cap)) / (-threshold * cap - (-max)) * (value - (-max)) + (-cap)
    
    def calc_fe(self, exp, rotate_pitch, rotate_yaw, rotate_roll, 
                eyes:float=0, 
                eyebrow:float=0, 
                wink:float=0, 
                pupil_x:float=0, 
                pupil_y:float=0, 
                mouth:float=0, 
                eee:float=0, 
                woo:float=0,
                smile:float=0):

        exp[0, 20, 1] += smile * -0.01
        exp[0, 14, 1] += smile * -0.02
        exp[0, 17, 1] += smile * 0.0065
        exp[0, 17, 2] += smile * 0.003
        exp[0, 13, 1] += smile * -0.00275
        exp[0, 16, 1] += smile * -0.00275
        exp[0, 3, 1] += smile * -0.0035
        exp[0, 7, 1] += smile * -0.0035

        exp[0, 19, 1] += mouth * 0.001
        exp[0, 19, 2] += mouth * 0.0001
        exp[0, 17, 1] += mouth * -0.0001
        rotate_pitch -= mouth * 0.05

        exp[0, 20, 2] += eee * -0.001
        exp[0, 20, 1] += eee * -0.001
        #x_d_new[0, 19, 1] += eee * 0.0006
        exp[0, 14, 1] += eee * -0.001

        exp[0, 14, 1] += woo * 0.001
        exp[0, 3, 1] += woo * -0.0005
        exp[0, 7, 1] += woo * -0.0005
        exp[0, 17, 2] += woo * -0.0005

        exp[0, 11, 1] += wink * 0.001
        exp[0, 13, 1] += wink * -0.0003
        exp[0, 17, 0] += wink * 0.0003
        exp[0, 17, 1] += wink * 0.0003
        exp[0, 3, 1] += wink * -0.0003
        
        rotate_roll -= wink * 0.1
        rotate_yaw -= wink * 0.1

        if 0 < pupil_x:
            exp[0, 11, 0] += pupil_x * 0.0007
            exp[0, 15, 0] += pupil_x * 0.001
        else:
            exp[0, 11, 0] += pupil_x * 0.001
            exp[0, 15, 0] += pupil_x * 0.0007

        exp[0, 11, 1] += pupil_y * -0.001
        exp[0, 15, 1] += pupil_y * -0.001
        eyes -= pupil_y / 2.

        exp[0, 11, 1] += eyes * -0.001
        exp[0, 13, 1] += eyes * 0.0003
        exp[0, 15, 1] += eyes * -0.001
        exp[0, 16, 1] += eyes * 0.0003
        exp[0, 1, 1] += eyes * -0.00025
        exp[0, 2, 1] += eyes * 0.00025

        if 0 < eyebrow:
            exp[0, 1, 1] += eyebrow * 0.001
            exp[0, 2, 1] += eyebrow * -0.001
        else:
            exp[0, 1, 0] += eyebrow * -0.001
            exp[0, 2, 0] += eyebrow * 0.001
            exp[0, 1, 1] += eyebrow * 0.0003
            exp[0, 2, 1] += eyebrow * -0.0003

        return rotate_pitch, rotate_yaw, rotate_roll
    
    def generate(self, 
                 img_source : np.ndarray, 
                 img_driver : np.ndarray, 
                 expression_multiplier: float = 1.5,
                 rotation_cap_pitch: float = 45,
                 rotation_cap_yaw: float = 45,
                 rotation_cap_roll: float = 45,
                 driving_multiplier: float = 1.75,
                 stitching: bool = False):
        """

        arguments

         img_source             np.ndarray      HW HWC 1HWC   uint8/float32
         
         img_driver             np.ndarray      HW HWC 1HWC   uint8/float32
         
         driving_multiplier     float           driving multiplier
                  
         rotation_cap_pitch     float           rotation cap for pitch
         
         rotation_cap_yaw       float           rotation cap for yaw
         
         rotation_cap_roll      float           rotation cap for roll
        """
        
        DRIVER_SIZE = (256, 256)
        SOURCE_SIZE = (256, 256)
        
        if self.kp_info_s is None:
            ip_s = ImageProcessor(img_source[:,:,:3]).resize(SOURCE_SIZE, interpolation=ImageProcessor.Interpolation.LANCZOS4)
            i_s = ip_s.get_image('HWC')
            i_t_s = torch.from_numpy(i_s).permute(2, 0, 1).unsqueeze(0).to('cuda')
            i_t_s = (i_t_s / 255).to(torch.float16)
            self.kp_info_s = self.live_portrait_wrapper_animal.get_kp_info(i_t_s)
            self.f_s = self.live_portrait_wrapper_animal.extract_feature_3d(i_t_s)
            self.x_s = self.live_portrait_wrapper_animal.transform_keypoint(self.kp_info_s)
        
        kp_info_s = self.kp_info_s
        x_c_s = kp_info_s['kp']
        f_s = self.f_s
        x_s = self.x_s
        
        ip_d = ImageProcessor(img_driver).resize(DRIVER_SIZE, interpolation=ImageProcessor.Interpolation.LANCZOS4)
        i_d = ip_d.get_image('HWC')
        i_t_d = torch.from_numpy(i_d).permute(2, 0, 1).unsqueeze(0).to('cuda')
        i_t_d = (i_t_d / 255).to(torch.float16)
        kp_info_d = self.live_portrait_wrapper_animal.get_kp_info(i_t_d)
        
        # cap rotation
        kp_info_d['pitch'] = self.cap_value(kp_info_d['pitch'], rotation_cap_pitch, 180)
        kp_info_d['yaw'] = self.cap_value(kp_info_d['yaw'], rotation_cap_yaw, 180)
        kp_info_d['roll'] = self.cap_value(kp_info_d['roll'], rotation_cap_roll, 180)
        
        delta_new = kp_info_d['exp']
        
        # p, y, r = self.calc_fe(delta_new, kp_info_d['pitch'], kp_info_d['yaw'], kp_info_d['roll'])
        # R_d = get_rotation_matrix(kp_info_d['pitch'] + p, kp_info_d['yaw'] + y, kp_info_d['roll'] + r)
        
        R_d = get_rotation_matrix(kp_info_d['pitch'], kp_info_d['yaw'], kp_info_d['roll'])
        
        t_new = kp_info_d['t']
        t_new[..., 2].fill_(0)
        scale_new = kp_info_d['scale']
        
        x_d_i = scale_new * (x_c_s @ R_d + delta_new * expression_multiplier) + t_new
        
        if self.x_d_0 is None:
            self.x_d_0 = x_d_i
            self.motion_multiplier = calc_motion_multiplier(x_s, self.x_d_0)
        
        x_d_0 = self.x_d_0
        motion_multiplier = self.motion_multiplier
        
        x_d_diff = (x_d_i - x_d_0) * motion_multiplier
        x_d_i = x_d_diff + x_s
        
        # TODO: stiching
        if stitching:
            x_d_i = self.live_portrait_wrapper_animal.stitching(x_s, x_d_i)
        
        x_d_i = x_s + (x_d_i - x_s) * driving_multiplier
        
        out = self.live_portrait_wrapper_animal.warp_decode(f_s, x_s, x_d_i)
        I_p = self.live_portrait_wrapper_animal.parse_output(out['out'])[0] # HWC
        
        # TODO: paste back if flag pastback and do_crop and stitching
        
        # ip = ImageProcessor(img_source)
        # dtype = ip.get_dtype()
        # _,H,W,_ = ip.get_dims()

        # out = self._generator.run(['out'], {'in_src': ip.resize(self.get_input_size()).ch(3).swap_ch().to_ufloat32(as_tanh=True).get_image('NCHW'),
        #                                     'in_drv' : ImageProcessor(img_driver).resize(self.get_input_size()).ch(3).swap_ch().to_ufloat32(as_tanh=True).get_image('NCHW'),
        #                                     'in_drv_start_motion' : driver_start_motion,
        #                                     'in_power' : np.array([power], np.float32)
        #                                     })[0].transpose(0,2,3,1)[0]

        # out = ImageProcessor(out).to_dtype(dtype, from_tanh=True).resize((W,H)).swap_ch().get_image('HWC')
        
        return I_p

