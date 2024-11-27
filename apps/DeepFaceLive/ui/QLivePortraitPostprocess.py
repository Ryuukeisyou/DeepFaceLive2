from pathlib import Path

from localization import L
from resources.gfx import QXImageDB
from xlib import qt as qtx

from ..backend import LivePortraitPostprocess
from .widgets.QBackendPanel import QBackendPanel
from .widgets.QCheckBoxCSWFlag import QCheckBoxCSWFlag
from .widgets.QComboBoxCSWDynamicSingleSwitch import \
    QComboBoxCSWDynamicSingleSwitch
from .widgets.QLabelPopupInfo import QLabelPopupInfo
from .widgets.QSpinBoxCSWNumber import QSpinBoxCSWNumber

from .widgets.QXPushButtonCSWSignal import QXPushButtonCSWSignal
from .widgets.QSliderCSWNumber import QSliderCSWNumber

class QLivePortraitPostprocess(QBackendPanel):
    def __init__(self, backend : LivePortraitPostprocess):

        cs = backend.get_control_sheet()

        grid_l = qtx.QXGridLayout( spacing=5)
        row = 0
        # grid_l.addWidget(q_device_label, row, 0, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        # grid_l.addWidget(q_device, row, 1, alignment=qtx.AlignLeft )
        # row += 1

        super().__init__(backend, L('@QLivePortraitPostprocess.module_title'),
                         layout=qtx.QXVBoxLayout([grid_l]) )
