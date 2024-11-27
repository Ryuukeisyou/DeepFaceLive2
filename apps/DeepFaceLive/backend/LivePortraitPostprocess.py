import time
from pathlib import Path

import numpy as np
from modelhub.torch import LivePortrait
from xlib import cv as lib_cv2
from xlib import os as lib_os
from xlib import path as lib_path
from xlib.image.ImageProcessor import ImageProcessor
from xlib.mp import csw as lib_csw

from .BackendBase import (BackendConnection, BackendDB, BackendHost,
                          BackendSignal, BackendWeakHeap, BackendWorker,
                          BackendWorkerState)




class LivePortraitPostprocess(BackendHost):
    def __init__(self, weak_heap : BackendWeakHeap, reemit_frame_signal : BackendSignal, bc_in : BackendConnection, bc_out : BackendConnection, animatables_path : Path, backend_db : BackendDB = None,
                  id : int = 0):
        self._id = id
        super().__init__(backend_db=backend_db,
                         sheet_cls=Sheet,
                         worker_cls=LivePortraitPostprocessWorker,
                         worker_state_cls=WorkerState,
                         worker_start_args=[weak_heap, reemit_frame_signal, bc_in, bc_out, animatables_path])

    def get_control_sheet(self) -> 'Sheet.Host': return super().get_control_sheet()

    def _get_name(self):
        return super()._get_name()

class LivePortraitPostprocessWorker(BackendWorker):
    def get_state(self) -> 'WorkerState': return super().get_state()
    def get_control_sheet(self) -> 'Sheet.Worker': return super().get_control_sheet()

    def on_start(self, weak_heap : BackendWeakHeap, reemit_frame_signal : BackendSignal, bc_in : BackendConnection, bc_out : BackendConnection, animatables_path : Path):
        self.weak_heap = weak_heap
        self.reemit_frame_signal = reemit_frame_signal
        self.bc_in = bc_in
        self.bc_out = bc_out
        self.pending_bcd = None
        
        lib_os.set_timer_resolution(1)

        state, cs = self.get_state(), self.get_control_sheet()

        cs.stitching.call_on_flag(self.on_cs_stitching)
        cs.stitching.enable()
        cs.stitching.set_flag(state.stitching)

    def on_cs_stitching(self, stitching):
        self.live_portrait_model.clear_source_cache()
        state, cs = self.get_state(), self.get_control_sheet()
        cs.stitching.set_flag(stitching)
        state.stitching = stitching
        self.save_state()
        self.reemit_frame_signal.send()

    def on_tick(self):        
        state, cs = self.get_state(), self.get_control_sheet()

        if self.pending_bcd is None:
            self.start_profile_timing()

            bcd = self.bc_in.read(timeout=0.005)
            if bcd is not None:
                bcd.assign_weak_heap(self.weak_heap)
                
                for i, fsi in enumerate(bcd.get_face_swap_info_list()):
                    lp_raw_out = bcd.get_image(fsi.live_portrait_raw_out_name)
                    if lp_raw_out is not None:
                        out_img = (lp_raw_out * 255).astype(np.uint8)
                        lp_flag_source_changed = bcd.get_file('lp_flag_source_changed')
                        if lp_flag_source_changed == 1:
                            self.lp_M_c2o = np.fromfile(bcd.get_file('lp_M_c2o'))
                            self.lp_i_s_orig = np.fromfile(bcd.get_file('lp_i_s_orig'))
                            self.lp_mask_ori_float = np.fromfile(bcd.get_file('lp_mask_ori_float'))
                        stiching = int(bcd.get_file('lp_stitching'))
                        if stiching == 1:
                            out_img = LivePortrait.paste_back(out_img, self.lp_M_c2o, self.lp_i_s_orig, self.lp_mask_ori_float)
                        
                        out_img = ImageProcessor(out_img).get_image('HWC')
                        fsi.face_swap_image_name = f'{fsi.live_portrait_raw_out_name}_swapped'
                        bcd.set_image(fsi.face_swap_image_name, out_img)
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
            self.stitching = lib_csw.Flag.Client()

    class Worker(lib_csw.Sheet.Worker):
        def __init__(self):
            super().__init__()
            self.stitching = lib_csw.Flag.Host()

class WorkerState(BackendWorkerState):
    stitching: bool = None
