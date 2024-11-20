from localization import L
from xlib import qt as qtx

from .widgets.QBackendPanel import QBackendPanel
from .widgets.QComboBoxCSWDynamicSingleSwitch import \
    QComboBoxCSWDynamicSingleSwitch
from .widgets.QLabelPopupInfo import QLabelPopupInfo
from .widgets.QSpinBoxCSWNumber import QSpinBoxCSWNumber


class QFaceCropper(QBackendPanel):
    def __init__(self, backend):
        cs = backend.get_control_sheet()

        q_device_label       = QLabelPopupInfo(label=L('@common.device'), popup_info_text=L('@common.help.device') )
        q_device             = QComboBoxCSWDynamicSingleSwitch(cs.device, reflect_state_widgets=[q_device_label])

        q_coverage_label = QLabelPopupInfo(label=L('@QFaceCropper.coverage'), popup_info_text=L('@QFaceCropper.help.coverage') )
        q_coverage       = QSpinBoxCSWNumber(cs.coverage, reflect_state_widgets=[q_coverage_label])

        grid_l = qtx.QXGridLayout(spacing=5)
        row = 0
        grid_l.addWidget(q_device_label, row, 0, 1, 1, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        grid_l.addWidget(q_device, row, 1, 1, 3 )
        row += 1

        sub_row = 0
        sub_grid_l = qtx.QXGridLayout(spacing=5)
        sub_grid_l.addWidget(q_coverage_label, sub_row, 0, 1, 1, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        sub_grid_l.addWidget(q_coverage, sub_row, 1, 1, 1, alignment=qtx.AlignLeft )
        sub_row += 1

        grid_l.addLayout(sub_grid_l, row, 0, 1, 2, alignment=qtx.AlignCenter )
        row += 1

        super().__init__(backend, L('@QFaceCropper.module_title'),
                         layout=qtx.QXVBoxLayout([grid_l]) )








