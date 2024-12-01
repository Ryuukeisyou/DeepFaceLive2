import time
from pathlib import Path

import numpy as np
from modelhub.torch import LivePortrait
from modelhub.torch import FasterLivePortrait
from xlib import cv as lib_cv2
from xlib import os as lib_os
from xlib import path as lib_path
from xlib.image.ImageProcessor import ImageProcessor
from xlib.mp import csw as lib_csw

from .BackendBase import (BackendConnection, BackendDB, BackendHost,
                          BackendSignal, BackendWeakHeap, BackendWorker,
                          BackendWorkerState)


class LivePortraitAnimator(BackendHost):
    def __init__(self, weak_heap : BackendWeakHeap, reemit_frame_signal : BackendSignal, bc_in : BackendConnection, bc_out : BackendConnection, animatables_path : Path, backend_db : BackendDB = None,
                  id : int = 0):
        self._id = id
        super().__init__(backend_db=backend_db,
                         sheet_cls=Sheet,
                         worker_cls=LivePortraitAnimatorWorker,
                         worker_state_cls=WorkerState,
                         worker_start_args=[weak_heap, reemit_frame_signal, bc_in, bc_out, animatables_path])

    def get_control_sheet(self) -> 'Sheet.Host': return super().get_control_sheet()

    def _get_name(self):
        return super()._get_name()

class LivePortraitAnimatorWorker(BackendWorker):
    def get_state(self) -> 'WorkerState': return super().get_state()
    def get_control_sheet(self) -> 'Sheet.Worker': return super().get_control_sheet()

    def on_start(self, weak_heap : BackendWeakHeap, reemit_frame_signal : BackendSignal, bc_in : BackendConnection, bc_out : BackendConnection, animatables_path : Path):
        self.weak_heap = weak_heap
        self.reemit_frame_signal = reemit_frame_signal
        self.bc_in = bc_in
        self.bc_out = bc_out
        self.animatables_path = animatables_path

        self.pending_bcd = None

        self.live_portrait : FasterLivePortrait = None

        self.animatable_img = None

        lib_os.set_timer_resolution(1)

        state, cs = self.get_state(), self.get_control_sheet()

        cs.device.call_on_selected(self.on_cs_device)
        cs.is_animal.call_on_selected(self.on_cs_is_animal)
        cs.animatable.call_on_selected(self.on_cs_animatable)

        cs.animator_face_id.call_on_number(self.on_cs_animator_face_id)
        cs.expression_multiplier.call_on_number(self.on_cs_expression_multiplier)
        cs.rotation_multiplier.call_on_number(self.on_cs_rotation_multiplier)
        cs.translation_multiplier.call_on_number(self.on_cs_translation_multiplier)
        cs.driving_multiplier.call_on_number(self.on_cs_driving_multiplier)
        cs.retarget_eye.call_on_number(self.on_cs_retarget_eye)
        cs.rotation_cap_pitch.call_on_number(self.on_cs_rotation_cap_pitch)
        cs.rotation_cap_yaw.call_on_number(self.on_cs_rotation_cap_yaw)
        cs.rotation_cap_roll.call_on_number(self.on_cs_rotation_cap_roll)
        cs.update_animatables.call_on_signal(self.update_animatables)
        cs.stitching.call_on_flag(self.on_cs_stitching)
        cs.reset_reference_pose.call_on_signal(self.on_cs_reset_reference_pose)

        cs.device.enable()
        cs.device.set_choices( FasterLivePortrait.get_available_devices(), none_choice_name='@misc.menu_select')
        cs.device.select(state.device)
        
        cs.is_animal.enable()
        cs.is_animal.set_choices([True, False], none_choice_name='@misc.menu_select')
        cs.is_animal.select(state.is_animal)

    def update_animatables(self):
        state, cs = self.get_state(), self.get_control_sheet()
        cs.animatable.set_choices([animatable_path.name for animatable_path in lib_path.get_files_paths(self.animatables_path, extensions=['.jpg','.jpeg','.png'])], none_choice_name='@misc.menu_select')


    def on_cs_device(self, idx, device):
        state, cs = self.get_state(), self.get_control_sheet()
        if device is None:
            self.terminate_flp(cs)
            state.device = device
            self.save_state()     
        else:
            if state.device == device:
                if self.live_portrait is None and state.is_animal is not None:
                    self.initialize_flp(state, cs)
            else:
                state.device = device
                self.save_state()          
                if self.live_portrait is None:
                    if state.is_animal is not None:
                        self.initialize_flp(state, cs)
                    else:
                        pass
                else:
                    self.terminate_flp(cs)
                    self.initialize_flp(state, cs)
    
    def on_cs_is_animal(self, idx, is_animal):
        state, cs = self.get_state(), self.get_control_sheet()
        if is_animal is None:
            self.terminate_flp(cs)
            state.is_animal = is_animal
            self.save_state()     
        else:
            if state.is_animal == is_animal:
                if self.live_portrait is None and state.device is not None:
                    self.initialize_flp(state, cs)
            else:
                state.is_animal = is_animal
                self.save_state()          
                if self.live_portrait is None:
                    if state.device is not None:
                        self.initialize_flp(state, cs)
                    else:
                        pass
                else:
                    self.terminate_flp(cs)
                    self.initialize_flp(state, cs)
       
    def initialize_flp(self, state:'WorkerState', cs:'Sheet.Worker'):
        self.live_portrait = FasterLivePortrait(device_info=state.device, engine='trt', is_animal=state.is_animal)
        cs.animatable.enable()
        self.update_animatables()
        cs.animatable.select(state.animatable)

        cs.animator_face_id.enable()
        cs.animator_face_id.set_config(lib_csw.Number.Config(min=0, max=16, step=1, decimals=0, allow_instant_update=True))
        cs.animator_face_id.set_number(state.animator_face_id if state.animator_face_id is not None else 0)
        
        cs.expression_multiplier.enable()
        cs.expression_multiplier.set_config(lib_csw.Number.Config(min=0.0, max=4.0, step=0.01, decimals=2, allow_instant_update=True))
        cs.expression_multiplier.set_number(state.expression_multiplier if state.expression_multiplier is not None else 1.0)

        cs.rotation_multiplier.enable()
        cs.rotation_multiplier.set_config(lib_csw.Number.Config(min=0.0, max=2.0, step=0.01, decimals=2, allow_instant_update=True))
        cs.rotation_multiplier.set_number(state.rotation_multiplier if state.rotation_multiplier is not None else 1.0)

        cs.translation_multiplier.enable()
        cs.translation_multiplier.set_config(lib_csw.Number.Config(min=0.0, max=2.0, step=0.01, decimals=2, allow_instant_update=True))
        cs.translation_multiplier.set_number(state.translation_multiplier if state.translation_multiplier is not None else 1.0)

        cs.driving_multiplier.enable()
        cs.driving_multiplier.set_config(lib_csw.Number.Config(min=0.0, max=2.0, step=0.01, decimals=2, allow_instant_update=True))
        cs.driving_multiplier.set_number(state.driving_multiplier if state.driving_multiplier is not None else 1.0)

        cs.retarget_eye.enable()
        cs.retarget_eye.set_config(lib_csw.Number.Config(min=0.0, max=2.0, step=0.05, decimals=2, allow_instant_update=True))
        cs.retarget_eye.set_number(state.retarget_eye if state.retarget_eye is not None else 1.0)

        cs.rotation_cap_pitch.enable()
        cs.rotation_cap_pitch.set_config(lib_csw.Number.Config(min=0.0, max=90, step=1, decimals=2, allow_instant_update=True))
        cs.rotation_cap_pitch.set_number(state.rotation_cap_pitch if state.rotation_cap_pitch is not None else 45)
        
        cs.rotation_cap_yaw.enable()
        cs.rotation_cap_yaw.set_config(lib_csw.Number.Config(min=0.0, max=90, step=1, decimals=2, allow_instant_update=True))
        cs.rotation_cap_yaw.set_number(state.rotation_cap_yaw if state.rotation_cap_yaw is not None else 45)

        cs.rotation_cap_roll.enable()
        cs.rotation_cap_roll.set_config(lib_csw.Number.Config(min=0.0, max=90, step=1, decimals=2, allow_instant_update=True))
        cs.rotation_cap_roll.set_number(state.rotation_cap_roll if state.rotation_cap_roll is not None else 45)
        
        cs.stitching.enable()
        cs.stitching.set_flag(state.stitching if state.stitching is not None else False)

        cs.update_animatables.enable()
        cs.reset_reference_pose.enable()

    def terminate_flp(self, cs:'Sheet.Worker'):
        self.live_portrait = None
        
        cs.animatable.disable()

        cs.animator_face_id.disable()
        
        cs.expression_multiplier.disable()

        cs.rotation_multiplier.disable()

        cs.translation_multiplier.disable()

        cs.driving_multiplier.disable()

        cs.rotation_cap_pitch.disable()
        
        cs.rotation_cap_yaw.disable()

        cs.rotation_cap_roll.disable()
        
        cs.stitching.disable()

        cs.update_animatables.disable()

        cs.reset_reference_pose.disable()

    def on_cs_animatable(self, idx, animatable):
        state, cs = self.get_state(), self.get_control_sheet()

        state.animatable = animatable
        self.animatable_img = None
        self.live_portrait.clear_ref_motion_cache()
        self.live_portrait.clear_source_cache()

        if animatable is not None:
            try:
                ip = ImageProcessor(lib_cv2.imread(self.animatables_path / animatable))
                self.animatable_img = ip.get_image('HWC')
            except Exception as e:
                cs.animatable.unselect()

        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_animator_face_id(self, animator_face_id):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.animator_face_id.get_config()
        animator_face_id = state.animator_face_id = int(np.clip(animator_face_id, cfg.min, cfg.max))
        cs.animator_face_id.set_number(animator_face_id)
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_expression_multiplier(self, expression_multiplier):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.expression_multiplier.get_config()
        expression_multiplier = state.expression_multiplier = float(np.clip(expression_multiplier, cfg.min, cfg.max))
        cs.expression_multiplier.set_number(expression_multiplier)
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_rotation_multiplier(self, rotation_multiplier):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.rotation_multiplier.get_config()
        rotation_multiplier = state.rotation_multiplier = float(np.clip(rotation_multiplier, cfg.min, cfg.max))
        cs.rotation_multiplier.set_number(rotation_multiplier)
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_translation_multiplier(self, translation_multiplier):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.translation_multiplier.get_config()
        translation_multiplier = state.translation_multiplier = float(np.clip(translation_multiplier, cfg.min, cfg.max))
        cs.translation_multiplier.set_number(translation_multiplier)
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_driving_multiplier(self, driving_multiplier):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.driving_multiplier.get_config()
        driving_multiplier = state.driving_multiplier = float(np.clip(driving_multiplier, cfg.min, cfg.max))
        cs.driving_multiplier.set_number(driving_multiplier)
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_retarget_eye(self, retarget_eye):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.retarget_eye.get_config()
        retarget_eye = state.retarget_eye = float(np.clip(retarget_eye, cfg.min, cfg.max))
        cs.retarget_eye.set_number(retarget_eye)
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_rotation_cap_pitch(self, rotation_cap_pitch):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.rotation_cap_pitch.get_config()
        rotation_cap_pitch = state.rotation_cap_pitch = float(np.clip(rotation_cap_pitch, cfg.min, cfg.max))
        cs.rotation_cap_pitch.set_number(rotation_cap_pitch)
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_rotation_cap_yaw(self, rotation_cap_yaw):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.rotation_cap_yaw.get_config()
        rotation_cap_yaw = state.rotation_cap_yaw = float(np.clip(rotation_cap_yaw, cfg.min, cfg.max))
        cs.rotation_cap_yaw.set_number(rotation_cap_yaw)
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_rotation_cap_roll(self, rotation_cap_roll):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.rotation_cap_roll.get_config()
        rotation_cap_roll = state.rotation_cap_roll = float(np.clip(rotation_cap_roll, cfg.min, cfg.max))
        cs.rotation_cap_roll.set_number(rotation_cap_roll)
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_stitching(self, stitching):
        self.live_portrait.clear_source_cache()
        state, cs = self.get_state(), self.get_control_sheet()
        cs.stitching.set_flag(stitching)
        state.stitching = stitching
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_reset_reference_pose(self):
        self.live_portrait.clear_ref_motion_cache()
        self.reemit_frame_signal.send()

    def on_tick(self):        
        state, cs = self.get_state(), self.get_control_sheet()

        if self.pending_bcd is None:
            self.start_profile_timing()

            bcd = self.bc_in.read(timeout=0.005)
            if bcd is not None:
                bcd.assign_weak_heap(self.weak_heap)

                lp = self.live_portrait
                if lp is not None and self.animatable_img is not None:

                    for i, fsi in enumerate(bcd.get_face_swap_info_list()):
                        if fsi.face_urect is not None and state.animator_face_id == i:
                            crop_image = bcd.get_image(fsi.face_crop_image_name)
                            
                            if crop_image is not None:

                                anim_image = lp.generate(
                                    self.animatable_img, 
                                    crop_image, 
                                    expression_multiplier=state.expression_multiplier,
                                    rotation_multiplier=state.rotation_multiplier,
                                    translation_multiplier=state.translation_multiplier,
                                    driving_multiplier=state.driving_multiplier, 
                                    retarget_eye=state.retarget_eye,
                                    rotation_cap_pitch=state.rotation_cap_pitch,
                                    rotation_cap_yaw=state.rotation_cap_yaw,
                                    rotation_cap_roll=state.rotation_cap_roll,
                                    stitching=state.stitching)
                                if anim_image is not None:
                                    anim_image = ImageProcessor(anim_image).get_image('HWC')
                           
                            else:
                                anim_image = ImageProcessor(self.animatable_img).get_image('HWC')
                            
                            if anim_image is not None:
                                fsi.face_swap_image_name = f'{fsi.image_name}_swapped'
                                bcd.set_image(fsi.face_swap_image_name, anim_image)
                                
                            break

                self.stop_profile_timing()
                self.pending_bcd = bcd

        if self.pending_bcd is not None:
            if self.bc_out.is_full_read(1):
                self.bc_out.write(self.pending_bcd)
                self.pending_bcd = None
            else:
                time.sleep(0.001)

class Sheet:
    class Host(lib_csw.Sheet.Host):
        def __init__(self):
            super().__init__()
            self.device = lib_csw.DynamicSingleSwitch.Client()
            self.is_animal = lib_csw.DynamicSingleSwitch.Client()
            self.animatable = lib_csw.DynamicSingleSwitch.Client()
            self.animator_face_id = lib_csw.Number.Client()
            self.update_animatables = lib_csw.Signal.Client()
            self.reset_reference_pose = lib_csw.Signal.Client()
            self.expression_multiplier = lib_csw.Number.Client()
            self.rotation_multiplier = lib_csw.Number.Client()
            self.translation_multiplier = lib_csw.Number.Client()
            self.driving_multiplier = lib_csw.Number.Client()
            self.retarget_eye = lib_csw.Number.Client()
            self.rotation_cap_pitch = lib_csw.Number.Client()
            self.rotation_cap_yaw= lib_csw.Number.Client()
            self.rotation_cap_roll = lib_csw.Number.Client()
            self.stitching = lib_csw.Flag.Client()

    class Worker(lib_csw.Sheet.Worker):
        def __init__(self):
            super().__init__()
            self.device = lib_csw.DynamicSingleSwitch.Host()
            self.is_animal = lib_csw.DynamicSingleSwitch.Host()
            self.animatable = lib_csw.DynamicSingleSwitch.Host()
            self.animator_face_id = lib_csw.Number.Host()
            self.update_animatables = lib_csw.Signal.Host()
            self.reset_reference_pose = lib_csw.Signal.Host()
            self.expression_multiplier = lib_csw.Number.Host()
            self.rotation_multiplier = lib_csw.Number.Host()
            self.translation_multiplier = lib_csw.Number.Host()
            self.driving_multiplier = lib_csw.Number.Host()
            self.retarget_eye = lib_csw.Number.Host()
            self.rotation_cap_pitch = lib_csw.Number.Host()
            self.rotation_cap_yaw = lib_csw.Number.Host()
            self.rotation_cap_roll = lib_csw.Number.Host()
            self.stitching = lib_csw.Flag.Host()

class WorkerState(BackendWorkerState):
    device = None
    is_animal : bool = None
    animatable : str = None
    animator_face_id : int = None
    expression_multiplier : float = None
    rotation_multiplier : float = None
    translation_multiplier: float = None
    driving_multiplier : float = None
    retarget_eye : float = None
    rotation_cap_pitch: float = None
    rotation_cap_yaw: float = None
    rotation_cap_roll: float = None
    stitching: bool = None
