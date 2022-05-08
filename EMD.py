from signalProcessingUtils import delay_filter, unit_filter


class EMD:
    def __init__(self, **kwargs):
        self.lpf = kwargs.setdefault('lpf', delay_filter)
        self.hpf = kwargs.setdefault('hpf', unit_filter)

    def forward(self, signal1, signal2):
        return self.lpf(signal1) * self.hpf(signal2) - self.lpf(signal2) * self.hpf(signal1)


# emd_test = EMD()
# s1 = (1, 2, 3)
# s2 = (2, 3, 4)
# emd_test.forward(s1, s2)
# print(emd_test.forward(s1, s2))
