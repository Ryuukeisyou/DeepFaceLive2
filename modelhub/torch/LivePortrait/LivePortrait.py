from pathlib import Path
import time
from typing import List

import numpy as np
import torch
from torch import Tensor

from xlib.file import SplittedFile
from xlib.image import ImageProcessor
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)

import os
import sys
repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
sys.path.append(repo_path)

from .repo.src.config.argument_config import ArgumentConfig
from .repo.src.config.inference_config import InferenceConfig
from .repo.src.config.crop_config import CropConfig
from .repo.src.live_portrait_wrapper import LivePortraitWrapperAnimal
from .repo.src.utils.camera import get_rotation_matrix
from .repo.src.utils.helper import calc_motion_multiplier
from .repo.src.utils.cropper import Cropper
from .repo.src.utils.io import resize_to_limit
from .repo.src.utils.crop import prepare_paste_back, paste_back
from .repo.src.utils.filter import smooth

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

        device_id = device_id=device_info.get_index()
        inference_cfg = InferenceConfig(device_id=device_id, flag_do_torch_compile=False)
        crop_cfg = CropConfig(device_id=device_id, scale=2.6, scale_crop_driving_video=2.4)
        self.live_portrait_wrapper_animal = LivePortraitWrapperAnimal(inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg, image_type='animal_face', flag_use_half_precision=inference_cfg.flag_use_half_precision)
        self.device = self.live_portrait_wrapper_animal.device
        
        self.t_pre = 0
        self.t_drive = 0
        self.t_kalman = 0
        self.t_warp_decode = 0
        self.t_post = 0
        self.t_pre_count = 0
        self.t_drive_count = 0
        self.t_kalman_count = 0
        self.t_warp_decode_count = 0
        self.t_post_count = 0
        
        self.i_s_orig = None
        self.crop_info = None
        self.mask_ori_float = None
        
        self.x_s_info = None
        self.f_s = None
        self.x_s = None
        
        self.x_d_0 = None
        self.motion_multiplier = None
        
        self.x_d_list = []


    def get_input_size(self):
        """
        returns optimal (Width,Height) for input images, thus you can resize source image to avoid extra load
        """
        return (256,256)
    
    def clear_source_cache(self):
        self.x_s_info = None

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
    
    def generate(self, 
                 img_source : np.ndarray, 
                 img_driver : np.ndarray, 
                 is_image: bool = True,
                 max_dim: int = 720,
                 expression_multiplier: float = 1.5,
                 rotation_multiplier: float = 1,
                 translation_multiplier: float = 1,
                 driving_multiplier: float = 1.75,
                 rotation_cap_pitch: float = 45,
                 rotation_cap_yaw: float = 45,
                 rotation_cap_roll: float = 45,
                 do_crop: bool = True,
                 stitching: bool = False,
                 pasteback: bool = True):
        """

        arguments

         img_source             np.ndarray      HW HWC 1HWC   uint8/float32
         
         img_driver             np.ndarray      HW HWC 1HWC   uint8/float32
         
         is_image               bool            source is image or video
         
         max_dim                int             input max dimension
         
         expression_multiplier  float           expression multiplier
         
         rotation_multiplier    float           rotation multiplier
         
         translation_multipler  float           translation multiplier
         
         driving_multiplier     float           driving multiplier
                  
         rotation_cap_pitch     float           rotation cap for pitch
         
         rotation_cap_yaw       float           rotation cap for yaw
         
         rotation_cap_roll      float           rotation cap for roll
        
        """
        
        DRIVER_SIZE = (256, 256)
        SOURCE_SIZE = (256, 256)
        
        if self.x_s_info is None or not is_image:
            i_s_orig = resize_to_limit(img_source[:,:,:3], max_dim)
            if do_crop:
                crop_info = self.cropper.crop_source_image(i_s_orig, self.cropper.crop_cfg)
                self.crop_info = crop_info
                self.i_s_orig = i_s_orig
                i_s = crop_info['img_crop_256x256']
                # prepare pasteback
                if pasteback and stitching:
                    mask_ori_float = prepare_paste_back(
                        self.live_portrait_wrapper_animal.inference_cfg.mask_crop, 
                        crop_info['M_c2o'], 
                        dsize=(i_s_orig.shape[1], i_s_orig.shape[0]))
                    self.mask_ori_float = mask_ori_float
            else:
                ip_s = ImageProcessor(img_source[:,:,:3]).resize(SOURCE_SIZE, interpolation=ImageProcessor.Interpolation.LANCZOS4)
                i_s = ip_s.get_image('HWC')
            self.i_s = i_s
            i_t_s = torch.from_numpy(i_s).permute(2, 0, 1).unsqueeze(0).to(self.device)
            i_t_s = (i_t_s / 255).to(torch.float16)
            self.x_s_info = self.live_portrait_wrapper_animal.get_kp_info(i_t_s)
            self.f_s = self.live_portrait_wrapper_animal.extract_feature_3d(i_t_s)
            self.x_s = self.live_portrait_wrapper_animal.transform_keypoint(self.x_s_info)
        
        i_s = self.i_s
        x_c_s = self.x_s_info['kp']
        f_s = self.f_s
        x_s = self.x_s


        # preprocess driving image
        t0 = time.perf_counter()
        ip_d = ImageProcessor(img_driver).resize(DRIVER_SIZE, interpolation=ImageProcessor.Interpolation.LANCZOS4)
        i_d = ip_d.get_image('HWC')
        i_t_d = torch.from_numpy(i_d).permute(2, 0, 1).unsqueeze(0).to(self.device)
        i_t_d = (i_t_d / 255).to(torch.float16)
        t1 = time.perf_counter()
        self.t_pre += t1 - t0
        self.t_pre_count += 1
        if self.t_pre_count == 100:
            print(f'[INFO] average preprocess time {self.t_pre / self.t_pre_count * 1000 :.2f}')
            self.t_pre = 0
            self.t_pre_count = 0
        
        
        # prepare driving data
        t0 = time.perf_counter()
        
        # compute driving key point info
        xs_info = self.live_portrait_wrapper_animal.get_kp_info(i_t_d)
        
        # cap and scale rotation angle
        xs_info['pitch'] = self.cap_value(xs_info['pitch'], rotation_cap_pitch, 180) * rotation_multiplier
        xs_info['yaw'] = self.cap_value(xs_info['yaw'], rotation_cap_yaw, 180) * rotation_multiplier
        xs_info['roll'] = self.cap_value(xs_info['roll'], rotation_cap_roll, 180) * rotation_multiplier
        
        # expression
        delta_new = xs_info['exp']
        
        # rotaion
        R_d = get_rotation_matrix(xs_info['pitch'], xs_info['yaw'], xs_info['roll'])
        
        # transform
        t_new = xs_info['t']
        t_new[..., 2].fill_(0)
        
        # scale
        scale_new = xs_info['scale']
        
        # driving data
        x_d_i = scale_new * (x_c_s @ R_d + delta_new * expression_multiplier) + t_new * translation_multiplier
        
        # save first driving data
        if self.x_d_0 is None:
            self.x_d_0 = x_d_i
            self.motion_multiplier = calc_motion_multiplier(x_s, self.x_d_0)
        
        # retrieve first driving data
        x_d_0 = self.x_d_0
        motion_multiplier = self.motion_multiplier
        
        # calculate difference
        x_d_diff = (x_d_i - x_d_0) * motion_multiplier
        x_d_i = x_d_diff + x_s
        
        # stiching
        if stitching:
            x_d_i = self.live_portrait_wrapper_animal.stitching(x_s, x_d_i)
        
        x_d_i = x_s + (x_d_i - x_s) * driving_multiplier
        
        t1 = time.perf_counter()
        self.t_drive += t1 - t0
        self.t_drive_count += 1
        if self.t_drive_count == 100:
            print(f'[INFO] average driving computation time {self.t_drive / self.t_drive_count * 1000 :.2f}')
            self.t_drive = 0
            self.t_drive_count = 0
        
        
        # kalman filter smoothing
        t0 = time.perf_counter()
        
        self.x_d_list.append(x_d_i)
        # buffer not enough
        if len(self.x_d_list) < 4:
            x_d_i = self.x_d_list[0].to(self.device)
        # buffer enough
        else:
            if len(self.x_d_list) == 4:
                self.x_d_list = smooth([i.cpu() for i in self.x_d_list], self.x_d_list[0].shape, device=self.device, observation_variance=3e-6)
                x_d_i = self.x_d_list[0].to(self.device)
            elif len(self.x_d_list) == 5:
                x_d_i = self.x_d_list[1].to(self.device)
            else:
                self.x_d_list = self.x_d_list[-4:]
                self.x_d_list = smooth([i.cpu() for i in self.x_d_list], self.x_d_list[0].shape, device=self.device, observation_variance=3e-6)
                x_d_i = self.x_d_list[0].to(self.device)
        
        t1 = time.perf_counter()
        self.t_kalman += t1 - t0
        self.t_kalman_count += 1
        if self.t_kalman_count == 100:
            print(f'[INFO] average kalman smooth time {self.t_kalman / self.t_kalman_count * 1000 :.2f}')
            self.t_kalman = 0
            self.t_kalman_count = 0
            
        
        # warp and decode
        t0 = time.perf_counter()
        
        out = self.live_portrait_wrapper_animal.warp_decode(f_s, x_s, x_d_i)
        
        t1 = time.perf_counter()
        self.t_warp_decode += t1 - t0
        self.t_warp_decode_count += 1
        if self.t_warp_decode_count == 100:
            print(f'[INFO] average warp decode time {self.t_warp_decode / self.t_warp_decode_count * 1000 :.2f}')
            self.t_warp_decode = 0
            self.t_warp_decode_count = 0
        
        
        # postprocessing
        t0 = time.perf_counter()
        
        I_p = self.live_portrait_wrapper_animal.parse_output(out['out'])[0] # HWC
              
        # paste back if flag pastback and do_crop and stitching
        if pasteback and do_crop and stitching:
            I_p_pstbk = paste_back(I_p, self.crop_info['M_c2o'], self.i_s_orig, self.mask_ori_float)
            I_p = I_p_pstbk
        
        t1 = time.perf_counter()
        self.t_post += t1 - t0
        self.t_post_count += 1
        if self.t_post_count == 100:
            print(f'[INFO] average postprocessing time {self.t_post / self.t_post_count * 1000 :.2f}')
            self.t_post = 0
            self.t_post_count = 0
        
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

