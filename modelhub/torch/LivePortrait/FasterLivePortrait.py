from pathlib import Path
import time
from typing import List

import copy
import cv2
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import Tensor

from xlib.file import SplittedFile
from xlib.image import ImageProcessor
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)

import os
import sys
repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faster_live_portrait")
sys.path.append(repo_path)

from .faster_live_portrait.src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from .faster_live_portrait.src.utils.crop import crop_image, paste_back_pytorch
from .faster_live_portrait.src.utils.utils import transform_keypoint, get_rotation_matrix, concat_feat, prepare_paste_back, calc_lip_close_ratio, calc_eye_close_ratio
from .faster_live_portrait.src.utils import utils

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

class FasterLivePortrait:
    """
    FasterLivePortrait: Bring portraits to life in Real Time!onnx/tensorrt support!
    https://github.com/warmshao/FasterLivePortrait

    arguments

     device_info    ORTDeviceInfo

        use FasterLivePortrait.get_available_devices()
        to determine a list of avaliable devices accepted by model
     
     engine         str

        "trt" or "onnx"
    
     is_animal      bool
     
        the source is animal or human
        
    raises
     Exception
    """
    
    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info(include_cpu=False)

    def __init__(self, device_info : ORTDeviceInfo, engine : str = "trt", is_animal : bool = False):
        if device_info not in FasterLivePortrait.get_available_devices():
            raise Exception(f'device_info {device_info} is not in available devices for LIA')

        match engine:
            case 'trt':
                cfg_path = os.path.join(repo_path, "configs", "trt_infer.yaml")
            case 'onnx':
                cfg_path = os.path.join(repo_path, "configs", "onnx_infer.yaml")
            case _:
                raise Exception("engine must be trt or onnx")
        
        self.device_id = device_info.get_index()
        
        cfg = OmegaConf.load(cfg_path)
        self.pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=is_animal, device=self.device_id)
        self.is_source_video = False      
        
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
    
    def clear_source_cache(self):
        self.pipe.src_infos.clear()

    def clear_ref_motion_cache(self):
        self.pipe.R_d_0 = None

    def calc_fe(self, 
                x_d_i_diff : Tensor, 
                rotate_pitch = 0, 
                rotate_yaw = 0,
                rotate_roll = 0,
                eyes = 1, 
                eyebrow = 1, 
                wink = 1, 
                pupil_x = 1, 
                pupil_y = 1, 
                mouth = 1, 
                eee = 1, 
                woo = 1, 
                smile = 1):

        x_d_i_diff[0, 20, 1] *= smile
        x_d_i_diff[0, 14, 1] *= smile
        x_d_i_diff[0, 17, 1] *= smile
        x_d_i_diff[0, 17, 2] *= smile
        x_d_i_diff[0, 13, 1] *= smile
        x_d_i_diff[0, 16, 1] *= smile
        x_d_i_diff[0, 3, 1] *= smile
        x_d_i_diff[0, 7, 1] *= smile

        x_d_i_diff[0, 19, 1] *= mouth
        x_d_i_diff[0, 19, 2] *= mouth
        x_d_i_diff[0, 17, 1] *= mouth
        rotate_pitch -= mouth * 0.05

        x_d_i_diff[0, 20, 2] *= eee
        x_d_i_diff[0, 20, 1] *= eee
        #x_d_new[0, 19, 1] *= eee
        x_d_i_diff[0, 14, 1] *= eee

        x_d_i_diff[0, 14, 1] *= woo
        x_d_i_diff[0, 3, 1] *= woo
        x_d_i_diff[0, 7, 1] *= woo
        x_d_i_diff[0, 17, 2] *= woo

        x_d_i_diff[0, 11, 1] *= wink
        x_d_i_diff[0, 13, 1] *= wink
        x_d_i_diff[0, 17, 0] *= wink
        x_d_i_diff[0, 17, 1] *= wink
        x_d_i_diff[0, 3, 1] *= wink
        rotate_roll -= wink * 0.1
        rotate_yaw -= wink * 0.1

        if 0 < pupil_x:
            x_d_i_diff[0, 11, 0] *= pupil_x
            x_d_i_diff[0, 15, 0] *= pupil_x
        else:
            x_d_i_diff[0, 11, 0] *= pupil_x
            x_d_i_diff[0, 15, 0] *= pupil_x

        x_d_i_diff[0, 11, 1] *= pupil_y
        x_d_i_diff[0, 15, 1] *= pupil_y
        eyes -= pupil_y / 2.

        x_d_i_diff[0, 11, 1] *= eyes
        x_d_i_diff[0, 13, 1] *= eyes
        x_d_i_diff[0, 15, 1] *= eyes
        x_d_i_diff[0, 16, 1] *= eyes
        x_d_i_diff[0, 1, 1] *= eyes
        x_d_i_diff[0, 2, 1] *= eyes

        if 0 < eyebrow:
            x_d_i_diff[0, 1, 1] *= eyebrow
            x_d_i_diff[0, 2, 1] *= eyebrow
        else:
            x_d_i_diff[0, 1, 0] *= eyebrow
            x_d_i_diff[0, 2, 0] *= eyebrow
            x_d_i_diff[0, 1, 1] *= eyebrow
            x_d_i_diff[0, 2, 1] *= eyebrow

        return rotate_pitch, rotate_yaw, rotate_roll

    def cap_value(self, value:np.ndarray, cap:float, max:float, threshold:float = 0.8):
        if (np.abs(value) < cap * threshold):
            return value
        elif np.abs(value) > max:
            return np.clamp(value, -cap, cap)
        else:
            if value > 0:
                return (cap - threshold * cap) / (max - threshold * cap) * (value - threshold * cap) + threshold * cap
            else:
                return (-threshold * cap - (-cap)) / (-threshold * cap - (-max)) * (value - (-max)) + (-cap)       
    
    def prepare_source(self, img_rgb):
        pipe = self.pipe
        pipe.src_infos = []
        
        lmk = None
        if pipe.is_animal:
            with torch.no_grad():
                img_rgb_pil = Image.fromarray(img_rgb)
                lmk = pipe.model_dict["xpose"].run(
                    img_rgb_pil,
                    'face',
                    'animal_face',
                    0,
                    0
                )
            if lmk is None:
                return
        else:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            src_faces = pipe.model_dict["face_analysis"].predict(img_bgr)
            if len(src_faces) == 0:
                print("No face detected in the this image.")
                return
            lmk = src_faces[0]
        
        # crop the face
        ret_dct = crop_image(
            img_rgb,  # ndarray
            lmk,  # 106x2 or Nx2
            dsize=pipe.cfg.crop_params.src_dsize,
            scale=pipe.cfg.crop_params.src_scale,
            vx_ratio=pipe.cfg.crop_params.src_vx_ratio,
            vy_ratio=pipe.cfg.crop_params.src_vy_ratio,
        )
        
        if pipe.is_animal:
            ret_dct["lmk_crop"] = lmk
        else:
            lmk = pipe.model_dict["landmark"].predict(img_rgb, lmk)
            ret_dct["lmk_crop"] = lmk
            ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / pipe.cfg.crop_params.src_dsize
        
        # update a 256x256 version for network input
        ret_dct["img_crop_256x256"] = cv2.resize(
            ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA
        )
        
        crop_info = ret_dct
        
        source_lmk = crop_info['lmk_crop']
        
        img_crop, img_crop_256x256 = crop_info['img_crop'], crop_info['img_crop_256x256']
        pitch, yaw, roll, t, exp, scale, kp = pipe.model_dict["motion_extractor"].predict(
            img_crop_256x256)
        x_s_info = {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "t": t,
            "exp": exp,
            "scale": scale,
            "kp": kp
        }
        
        src_info = []
        src_info.append(copy.deepcopy(x_s_info))
        x_c_s = kp
        R_s = get_rotation_matrix(pitch, yaw, roll)
        f_s = pipe.model_dict["app_feat_extractor"].predict(img_crop_256x256)
        x_s = transform_keypoint(pitch, yaw, roll, t, exp, scale, kp)
        
        src_info.extend([source_lmk.copy(), R_s.copy(), f_s.copy(), x_s.copy(), x_c_s.copy()])
        if not pipe.is_animal:
            flag_lip_zero = pipe.cfg.infer_params.flag_normalize_lip  # not overwrite
            if flag_lip_zero:
                # let lip-open scalar to be 0 at first
                c_d_lip_before_animation = [0.]
                combined_lip_ratio_tensor_before_animation = pipe.calc_combined_lip_ratio(
                    c_d_lip_before_animation, source_lmk)
                if combined_lip_ratio_tensor_before_animation[0][
                    0] < pipe.cfg.infer_params.lip_normalize_threshold:
                    flag_lip_zero = False
                    src_info.append(None)
                    src_info.append(flag_lip_zero)
                else:
                    lip_delta_before_animation = pipe.model_dict['stitching_lip_retarget'].predict(
                        concat_feat(x_s, combined_lip_ratio_tensor_before_animation))
                    src_info.append(lip_delta_before_animation.copy())
                    src_info.append(flag_lip_zero)
            else:
                src_info.append(None)
                src_info.append(flag_lip_zero)
        else:
            src_info.append(None)
            src_info.append(False)

        ######## prepare for pasteback ########
        if pipe.cfg.infer_params.flag_pasteback and pipe.cfg.infer_params.flag_do_crop and pipe.cfg.infer_params.flag_stitching:
            mask_ori_float = prepare_paste_back(pipe.mask_crop, crop_info['M_c2o'],
                                                dsize=(img_rgb.shape[1], img_rgb.shape[0]))
            mask_ori_float = torch.from_numpy(mask_ori_float).to(pipe.device)
            src_info.append(mask_ori_float)
        else:
            src_info.append(None)
        M = torch.from_numpy(crop_info['M_c2o']).to(pipe.device)
        src_info.append(M)
        
        img_source_t = torch.from_numpy(img_rgb).to(pipe.device).float()
        src_info.append(img_source_t)
        
        pipe.src_infos.append(src_info)
        
    def generate(self, 
                 img_source : np.ndarray, 
                 img_driver : np.ndarray, 
                 is_source_video: bool = False,
                 max_dim: int = 720,
                 src_scale: float = 2.6,
                 expression_multiplier: float = 1.5,
                 rotation_multiplier: float = 1,
                 translation_multiplier: float = 1,
                 driving_multiplier: float = 1.75,
                 rotation_cap_pitch: float = 45,
                 rotation_cap_yaw: float = 45,
                 rotation_cap_roll: float = 45,
                 do_crop: bool = True,
                 stitching: bool = False,
                 pasteback: bool = True) -> np.ndarray :
        """

        arguments

         img_source             np.ndarray      HW HWC 1HWC   uint8/float32
         
         img_driver             np.ndarray      HW HWC 1HWC   uint8/float32
         
         is_source_video        bool            source is image or video
         
         max_dim                int             input max dimension
         
         expression_multiplier  float           expression multiplier
         
         rotation_multiplier    float           rotation multiplier
         
         translation_multipler  float           translation multiplier
         
         driving_multiplier     float           driving multiplier
                  
         rotation_cap_pitch     float           rotation cap for pitch
         
         rotation_cap_yaw       float           rotation cap for yaw
         
         rotation_cap_roll      float           rotation cap for roll
        
        """
        pipe = self.pipe
        
        # set params
        pipe.is_source_video = is_source_video
        pipe.cfg.crop_params.src_scale = src_scale
        pipe.cfg.infer_params.source_max_dim = max_dim
        pipe.cfg.infer_params.flag_do_crop = do_crop
        pipe.cfg.infer_params.flag_stitching = stitching
        pipe.cfg.infer_params.flag_pasteback = pasteback
        pipe.cfg.infer_params.driving_multiplier = driving_multiplier
        
        img_rgb = img_driver
        if pipe.src_lmk_pre is None:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            if not pipe.is_animal:
                src_face = pipe.model_dict["face_analysis"].predict(img_bgr)
                if len(src_face) == 0:
                    pipe.src_lmk_pre = None
                    return None
                lmk = src_face[0]
            else:
                img_rgb_pil = Image.fromarray(img_rgb)
                lmk = pipe.model_dict["xpose"].run(
                    img_rgb_pil,
                    'face',
                    'animal_face',
                    0,
                    0
                )
            lmk = pipe.model_dict["landmark"].predict(img_rgb, lmk)
            pipe.src_lmk_pre = lmk.copy()
        else:
            lmk = pipe.model_dict["landmark"].predict(img_rgb, pipe.src_lmk_pre)
            pipe.src_lmk_pre = lmk.copy()
            
        lmk_crop = lmk.copy()
        img_crop = cv2.resize(img_rgb, (256, 256))

        input_eye_ratio = calc_eye_close_ratio(lmk_crop[None])
        input_lip_ratio = calc_lip_close_ratio(lmk_crop[None])
        pitch, yaw, roll, t, exp, scale, kp = pipe.model_dict["motion_extractor"].predict(img_crop)
        pitch = self.cap_value(pitch, rotation_cap_pitch, 180) * rotation_multiplier
        yaw = self.cap_value(yaw, rotation_cap_yaw, 180) * rotation_multiplier
        roll = self.cap_value(roll, rotation_cap_roll, 180) * rotation_multiplier
        x_d_i_info = {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "t": t * translation_multiplier,
            "exp": exp * expression_multiplier,
            "scale": scale,
            "kp": kp
        }
        R_d_i = get_rotation_matrix(pitch, yaw, roll)

        # save first driving frame data
        if pipe.R_d_0 is None:
            pipe.R_d_0 = R_d_i.copy()
            pipe.x_d_0_info = copy.deepcopy(x_d_i_info)
            # realtime smooth
            pipe.R_d_smooth = utils.OneEuroFilter(4, 1)
            pipe.exp_smooth = utils.OneEuroFilter(4, 1)
        R_d_0 = pipe.R_d_0.copy()
        x_d_0_info = copy.deepcopy(pipe.x_d_0_info)
        out_crop, out_org = None, None
        
        # prepare source if not exist
        if len(pipe.src_infos) == 0:
            self.prepare_source(img_source[:,:,:3])
        
        x_s_info, source_lmk, R_s, f_s, x_s, x_c_s, lip_delta_before_animation, flag_lip_zero, mask_ori_float, M, img_source_t = pipe.src_infos[0]
        if pipe.cfg.infer_params.flag_relative_motion:
            if pipe.is_source_video:
                if pipe.cfg.infer_params.flag_video_editing_head_rotation:
                    R_new = (R_d_i @ np.transpose(R_d_0, (0, 2, 1))) @ R_s
                    R_new = pipe.R_d_smooth.process(R_new)
                else:
                    R_new = R_s
            else:
                R_new = (R_d_i @ np.transpose(R_d_0, (0, 2, 1))) @ R_s
            delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
            if pipe.is_source_video:
                delta_new = pipe.exp_smooth.process(delta_new)
            scale_new = x_s_info['scale'] if pipe.is_source_video else x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
            t_new = x_s_info['t'] if pipe.is_source_video else x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
        else:
            if pipe.is_source_video:
                if pipe.cfg.infer_params.flag_video_editing_head_rotation:
                    R_new = R_d_i
                    R_new = pipe.R_d_smooth.process(R_new)
                else:
                    R_new = R_s
            else:
                R_new = R_d_i
            delta_new = x_d_i_info['exp'].copy()
            if pipe.is_source_video:
                delta_new = pipe.exp_smooth.process(delta_new)
            scale_new = x_s_info['scale'].copy()
            t_new = x_d_i_info['t'].copy()

        t_new[..., 2] = 0  # zero tz
        x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new
        if not pipe.is_animal:
            # Algorithm 1:
            if not pipe.cfg.infer_params.flag_stitching and not pipe.cfg.infer_params.flag_eye_retargeting and not pipe.cfg.infer_params.flag_lip_retargeting:
                # without stitching or retargeting
                if flag_lip_zero:
                    x_d_i_new += lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
                else:
                    pass
            elif pipe.cfg.infer_params.flag_stitching and not pipe.cfg.infer_params.flag_eye_retargeting and not pipe.cfg.infer_params.flag_lip_retargeting:
                # with stitching and without retargeting
                if flag_lip_zero:
                    x_d_i_new = pipe.stitching(x_s, x_d_i_new) + lip_delta_before_animation.reshape(
                        -1, x_s.shape[1], 3)
                else:
                    x_d_i_new = pipe.stitching(x_s, x_d_i_new)
            else:
                eyes_delta, lip_delta = None, None
                if pipe.cfg.infer_params.flag_eye_retargeting:
                    c_d_eyes_i = input_eye_ratio
                    combined_eye_ratio_tensor = pipe.calc_combined_eye_ratio(c_d_eyes_i,
                                                                                source_lmk)
                    # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                    eyes_delta = pipe.retarget_eye(x_s, combined_eye_ratio_tensor)
                if pipe.cfg.infer_params.flag_lip_retargeting:
                    c_d_lip_i = input_lip_ratio
                    combined_lip_ratio_tensor = pipe.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
                    # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                    lip_delta = pipe.retarget_lip(x_s, combined_lip_ratio_tensor)

                if pipe.cfg.infer_params.flag_relative_motion:  # use x_s
                    x_d_i_new = x_s + \
                                (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                                (lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)
                else:  # use x_d,i
                    x_d_i_new = x_d_i_new + \
                                (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                                (lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)

                if pipe.cfg.infer_params.flag_stitching:
                    x_d_i_new = pipe.stitching(x_s, x_d_i_new)
        else:
            if pipe.cfg.infer_params.flag_stitching:
                x_d_i_new = pipe.stitching(x_s, x_d_i_new)

        x_d_i_diff = x_d_i_new - x_s
        self.calc_fe(x_d_i_diff, eyes=2, pupil_x=1.1, pupil_y=1.1)
        x_d_i_new = x_s + x_d_i_diff * pipe.cfg.infer_params.driving_multiplier
        out_crop = pipe.model_dict["warping_spade"].predict(f_s, x_s, x_d_i_new)
        if pipe.cfg.infer_params.flag_pasteback and pipe.cfg.infer_params.flag_do_crop and pipe.cfg.infer_params.flag_stitching:
            # TODO: pasteback is slow, considering optimize it using multi-threading or GPU
            # I_p_pstbk = paste_back(out_crop, crop_info['M_c2o'], I_p_pstbk, mask_ori_float)
            I_p_pstbk = paste_back_pytorch(out_crop, M, img_source_t, mask_ori_float)
            return I_p_pstbk.to(dtype=torch.uint8).cpu().numpy()
        else:
            return out_crop.to(dtype=torch.uint8).cpu().numpy()

