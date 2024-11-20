import time
from enum import IntEnum
import numpy as np
from modelhub import onnx as onnx_models
from modelhub import cv as cv_models

from xlib import os as lib_os
from xlib.mp import csw as lib_csw

from .BackendBase import (BackendConnection, BackendDB, BackendHost,
                          BackendSignal, BackendWeakHeap, BackendWorker,
                          BackendWorkerState)


class FaceCropper(BackendHost):
    def __init__(self, weak_heap : BackendWeakHeap, reemit_frame_signal : BackendSignal, bc_in : BackendConnection, bc_out : BackendConnection, backend_db : BackendDB = None):

        super().__init__(backend_db=backend_db,
                         sheet_cls=Sheet,
                         worker_cls=FaceCropperWorker,
                         worker_state_cls=WorkerState,
                         worker_start_args=[weak_heap, reemit_frame_signal, bc_in, bc_out, ] )

    def get_control_sheet(self) -> 'Sheet.Host': return super().get_control_sheet()


class FaceCropperWorker(BackendWorker):
    def get_state(self) -> 'WorkerState': return super().get_state()
    def get_control_sheet(self) -> 'Sheet.Worker': return super().get_control_sheet()

    def on_start(self, weak_heap : BackendWeakHeap, reemit_frame_signal : BackendSignal,
                       bc_in : BackendConnection,
                       bc_out : BackendConnection,
                       ):
        self.weak_heap = weak_heap
        self.reemit_frame_signal = reemit_frame_signal
        self.bc_in = bc_in
        self.bc_out = bc_out
        self.pending_bcd = None

        lib_os.set_timer_resolution(1)

        state, cs = self.get_state(), self.get_control_sheet()
        cs.device.call_on_selected(self.on_cs_devices)
        cs.coverage.call_on_number(self.on_cs_coverage)
        
        cs.coverage.enable()
        cs.coverage.set_config(lib_csw.Number.Config(min=0.1, max=3.0, step=0.1, decimals=1, allow_instant_update=True))
        
        coverage = state.coverage
        if coverage is None:
            coverage = 1
        cs.coverage.set_number(coverage)
        
        cs.device.enable()
        cs.device.set_choices(['CPU'], none_choice_name='@misc.menu_select')
        cs.device.select(state.device)

    def on_cs_devices(self, idx, device):
        state, cs = self.get_state(), self.get_control_sheet()
        if device is not None and state.device == device:
            pass
        else:
            state.device = device
            self.save_state()
            self.restart()


    def on_cs_coverage(self, coverage):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.coverage.get_config()
        coverage = state.coverage = np.clip(coverage, cfg.min, cfg.max)
        cs.coverage.set_number(coverage)
        self.save_state()
        self.reemit_frame_signal.send()


    def on_tick(self):
        state, cs = self.get_state(), self.get_control_sheet()

        if self.pending_bcd is None and state.device is not None and state.coverage is not None:
            self.start_profile_timing()

            bcd = self.bc_in.read(timeout=0.005)
            if bcd is not None:
                bcd.assign_weak_heap(self.weak_heap)
                is_frame_reemitted = bcd.get_is_frame_reemitted()

                if state is not None:
                    frame_image_name = bcd.get_frame_image_name()
                    frame_image = bcd.get_image(frame_image_name)

                    if frame_image is not None:
                        fsi_list = bcd.get_face_swap_info_list()

                        for face_id, fsi in enumerate(fsi_list):
                            if fsi.face_urect is not None:
                                # Cut the face to feed to the face marker
                                face_image, face_uni_mat = fsi.face_urect.cut(frame_image, state.coverage, 256)
                                
                                fsi.image_to_align_uni_mat = face_uni_mat
                                fsi.face_crop_image_name = f'{frame_image_name}_{face_id}_crop'
                                bcd.set_image(fsi.face_crop_image_name, face_image)

                    self.stop_profile_timing()
                self.pending_bcd = bcd

        if self.pending_bcd is not None:
            if self.bc_out.is_full_read(1):
                self.bc_out.write(self.pending_bcd)
                self.pending_bcd = None
            else:
                time.sleep(0.001)

class WorkerState(BackendWorkerState):
    def __init__(self):
        self.coverage: float = None
        self.device = None

class Sheet:
    class Host(lib_csw.Sheet.Host):
        def __init__(self):
            super().__init__()
            self.device = lib_csw.DynamicSingleSwitch.Client()
            self.coverage = lib_csw.Number.Client()

    class Worker(lib_csw.Sheet.Worker):
        def __init__(self):
            super().__init__()
            self.device = lib_csw.DynamicSingleSwitch.Host()
            self.coverage = lib_csw.Number.Host()