from EOTF.SignalFilters import delay_filter, unit_filter
import numpy as np


class EMD:
    def __init__(self, **kwargs):
        self.lpf = kwargs.setdefault('lpf', delay_filter)
        self.hpf = kwargs.setdefault('hpf', unit_filter)

    def forward(self, signal1, signal2):
        return self.lpf(signal1) * self.hpf(signal2) - self.lpf(signal2) * self.hpf(signal1)

    def forward_row(self, buf, row_index):
        chosen_horizontal_line = np.array(buf)[:, row_index, :]
        return [self.forward(chosen_horizontal_line[:, i], chosen_horizontal_line[:, i + 1]) for i in
                range(chosen_horizontal_line.shape[1] - 1)]

    def forward_video(self, buffer):
        result = []
        for r in range(1, buffer[0].shape[0]):
            result.append(self.forward_row(buffer, r))
        return list(np.transpose(np.array(result), (2, 0, 1)))

    # TODO: Add kernel support (when time comes)
