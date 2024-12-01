from pathlib import Path

from localization import L
from resources.gfx import QXImageDB
from xlib import qt as qtx

from ..backend import LivePortraitAnimator
from .widgets.QBackendPanel import QBackendPanel
from .widgets.QCheckBoxCSWFlag import QCheckBoxCSWFlag
from .widgets.QComboBoxCSWDynamicSingleSwitch import \
    QComboBoxCSWDynamicSingleSwitch
from .widgets.QLabelPopupInfo import QLabelPopupInfo
from .widgets.QSpinBoxCSWNumber import QSpinBoxCSWNumber

from .widgets.QXPushButtonCSWSignal import QXPushButtonCSWSignal
from .widgets.QSliderCSWNumber import QSliderCSWNumber

class QLivePortraitAnimator(QBackendPanel):
    def __init__(self, backend : LivePortraitAnimator, animatables_path : Path):
        self._animatables_path = animatables_path

        cs = backend.get_control_sheet()

        btn_open_folder = self.btn_open_folder = qtx.QXPushButton(image = QXImageDB.eye_outline('light gray'), tooltip_text='Reveal in Explorer', released=self._btn_open_folder_released, fixed_size=(24,22) )

        q_device_label  = QLabelPopupInfo(label=L('@common.device'), popup_info_text=L('@common.help.device') )
        q_device        = QComboBoxCSWDynamicSingleSwitch(cs.device, reflect_state_widgets=[q_device_label])

        q_is_animal_label  = QLabelPopupInfo(label=L('@QLivePortraitAnimator.is_animal'))
        q_is_animal        = QComboBoxCSWDynamicSingleSwitch(cs.is_animal, reflect_state_widgets=[q_is_animal_label])

        q_animatable_label = QLabelPopupInfo(label=L('@QFaceAnimator.animatable') )
        q_animatable       = QComboBoxCSWDynamicSingleSwitch(cs.animatable, reflect_state_widgets=[q_animatable_label, btn_open_folder])

        q_update_animatables = QXPushButtonCSWSignal(cs.update_animatables, image=QXImageDB.reload_outline('light gray'), button_size=(24,22) )

        q_animator_face_id_label = QLabelPopupInfo(label=L('@QFaceAnimator.animator_face_id') )
        q_animator_face_id       = QSpinBoxCSWNumber(cs.animator_face_id, reflect_state_widgets=[q_animator_face_id_label])
        
        q_rotation_cap_pitch_label = QLabelPopupInfo(label=L('@QLivePortraitAnimator.rotation_cap_pitch'))
        q_rotation_cap_pitch       = QSpinBoxCSWNumber(cs.rotation_cap_pitch, reflect_state_widgets=[q_rotation_cap_pitch_label])
        q_rotation_cap_yaw_label   = QLabelPopupInfo(label=L('@QLivePortraitAnimator.rotation_cap_yaw'))
        q_rotation_cap_yaw      = QSpinBoxCSWNumber(cs.rotation_cap_yaw, reflect_state_widgets=[q_rotation_cap_yaw_label])
        q_rotation_cap_roll_label  = QLabelPopupInfo(label=L('@QLivePortraitAnimator.rotation_cap_roll'))
        q_rotation_cap_roll       = QSpinBoxCSWNumber(cs.rotation_cap_roll, reflect_state_widgets=[q_rotation_cap_roll_label])
        
        q_expression_multiplier_label = QLabelPopupInfo(label=L('@QLivePortraitAnimator.expression_multiplier') )
        q_expression_multiplier = QSliderCSWNumber(cs.expression_multiplier, reflect_state_widgets=[q_expression_multiplier_label])

        q_rotation_multiplier_label = QLabelPopupInfo(label=L('@QLivePortraitAnimator.rotation_multiplier') )
        q_rotation_multiplier = QSliderCSWNumber(cs.rotation_multiplier, reflect_state_widgets=[q_rotation_multiplier_label])

        q_translation_multiplier_label = QLabelPopupInfo(label=L('@QLivePortraitAnimator.translation_multiplier') )
        q_translation_multiplier = QSliderCSWNumber(cs.translation_multiplier, reflect_state_widgets=[q_translation_multiplier_label])

        q_driving_multiplier_label = QLabelPopupInfo(label=L('@QLivePortraitAnimator.driving_multiplier') )
        q_driving_multiplier = QSliderCSWNumber(cs.driving_multiplier, reflect_state_widgets=[q_driving_multiplier_label])

        q_retarget_eye_label = QLabelPopupInfo(label=L('@QLivePortraitAnimator.retarget_eye') )
        q_retarget_eye = QSpinBoxCSWNumber(cs.retarget_eye, reflect_state_widgets=[q_retarget_eye_label])

        q_stitching_label = QLabelPopupInfo(label=L('@QLivePortraitAnimator.stitching') )
        q_stitching = QCheckBoxCSWFlag(cs.stitching, reflect_state_widgets=q_stitching_label)
        
        q_reset_reference_pose = QXPushButtonCSWSignal(cs.reset_reference_pose, text=L('@QFaceAnimator.reset_reference_pose') )

        grid_l = qtx.QXGridLayout( spacing=5)
        row = 0
        grid_l.addWidget(q_device_label, row, 0, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        grid_l.addWidget(q_device, row, 1, alignment=qtx.AlignLeft )
        row += 1
        grid_l.addWidget(q_is_animal_label, row, 0, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        grid_l.addWidget(q_is_animal, row, 1, alignment=qtx.AlignLeft )
        row += 1
        grid_l.addWidget(q_animatable_label, row, 0, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        grid_l.addLayout(qtx.QXHBoxLayout([q_animatable, 2, btn_open_folder, 2, q_update_animatables]), row, 1 )
        row += 1
        grid_l.addWidget(q_animator_face_id_label, row, 0, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        grid_l.addWidget(q_animator_face_id, row, 1, alignment=qtx.AlignLeft )
        row += 1
        grid_l.addWidget(q_rotation_cap_pitch_label, row, 0, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        grid_l.addWidget(q_rotation_cap_pitch, row, 1 )
        row += 1
        grid_l.addWidget(q_rotation_cap_yaw_label, row, 0, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        grid_l.addWidget(q_rotation_cap_yaw, row, 1 )
        row += 1
        grid_l.addWidget(q_rotation_cap_roll_label, row, 0, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        grid_l.addWidget(q_rotation_cap_roll, row, 1 )
        row += 1
        grid_l.addWidget(q_expression_multiplier_label, row, 0, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        grid_l.addWidget(q_expression_multiplier, row, 1 )
        row += 1
        grid_l.addWidget(q_rotation_multiplier_label, row, 0, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        grid_l.addWidget(q_rotation_multiplier, row, 1 )
        row += 1
        grid_l.addWidget(q_translation_multiplier_label, row, 0, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        grid_l.addWidget(q_translation_multiplier, row, 1 )
        row += 1
        grid_l.addWidget(q_driving_multiplier_label, row, 0, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        grid_l.addWidget(q_driving_multiplier, row, 1 )
        row += 1
        grid_l.addWidget(q_retarget_eye_label, row, 0, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        grid_l.addWidget(q_retarget_eye, row, 1 )
        row += 1
        grid_l.addWidget(q_stitching_label, row, 0, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        grid_l.addWidget(q_stitching, row, 1 )
        row += 1
        
        grid_l.addWidget(q_reset_reference_pose, row, 0, 1, 2  )
        row += 1

        super().__init__(backend, L('@QLivePortraitAnimator.module_title'),
                         layout=qtx.QXVBoxLayout([grid_l]) )

    def _btn_open_folder_released(self):
        qtx.QDesktopServices.openUrl(qtx.QUrl.fromLocalFile( str(self._animatables_path) ))
