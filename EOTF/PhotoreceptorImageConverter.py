import math
import numpy as np


def make_gaussian_kernel(size: int, sigma: float = 1) -> np.array:
    coefficient = 1/((2 * np.pi) * (sigma ** 2))
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x * x + y * y)
    gaus = np.exp(-((d - 0) ** 2 / (2.0 * sigma ** 2)))
    return gaus / gaus.sum()


class PhotoreceptorImageConverter:
    def __init__(self, kernel: np.array, pic_shape: (int, int), photoreceptor_num: int, pad_mode = "symmetric"):
        # Assumption: conv1_kernel_size is an odd number
        self.kernel = kernel
        self.h_size = int(np.sqrt(pic_shape[1] * photoreceptor_num / pic_shape[0]).round())
        if self.h_size < 1: self.h_size = 1
        self.v_size = round(photoreceptor_num / self.h_size)
        self.h_stride = int(np.floor(pic_shape[1] / self.h_size))
        self.v_stride = int(np.floor(pic_shape[0] / self.v_size))
        self.h_padding = math.ceil((pic_shape[0] % self.h_stride) / 2)
        self.v_padding = math.ceil((pic_shape[1] % self.v_stride) / 2)
        self.pad_mode = pad_mode

    def apply(self, pic):
        if self.h_padding < 0 or self.v_padding < 0:
            padded_pic = self._slice_pic(pic)
        else:
            padded_pic = self._pad_input(pic)
        output = np.zeros((self.v_size, self.h_size))
        half_ker = (self.kernel.shape[0]-1)//2
        self._convolve(half_ker, output, padded_pic)
        return output

    def _slice_pic(self, pic):
        padded_pic = pic
        if self.v_padding < 0:
            padded_pic = padded_pic[-self.v_padding:self.v_padding, :]
            self.v_padding = 0
        if self.h_padding < 0:
            padded_pic = padded_pic[:, -self.h_padding:self.h_padding]
            self.h_padding = 0
        return padded_pic

    def _convolve(self, half_ker, output, padded_pic):
        for i in range(self.v_size):
            for j in range(self.h_size):
                input_row = i * self.v_stride + half_ker
                input_column = j * self.h_stride + half_ker
                output[i, j] = self._overlay_kernel(half_ker, input_row, input_column, padded_pic)

    def _overlay_kernel(self, half_ker, i, j, padded_pic):
        ker = self._kernel_sized_portion(padded_pic, half_ker, i, j)
        ker = np.pad(ker, ((0, self.kernel.shape[0] - ker.shape[0]), (0, self.kernel.shape[1] - ker.shape[1])))
        return np.multiply(self.kernel, ker).sum() / (self.kernel.shape[0] ** 2)

    @staticmethod
    def _kernel_sized_portion(pic, half_ker_size, center_row, center_col):
        return pic[center_row - half_ker_size:center_row + half_ker_size + 1,
                   center_col - half_ker_size:center_col + half_ker_size + 1]

    def _pad_input(self, pic: np.array) -> np.array:
#        padded_pic = np.zeros((pic.shape[0] + 2 * self.v_padding, pic.shape[1] + 2 * self.h_padding))
#        for i in range(self.v_padding, padded_pic.shape[0] - self.v_padding):
#            for j in range(self.h_padding, padded_pic.shape[1] - self.h_padding):
#                padded_pic[i, j] = pic[i - self.v_padding, j - self.h_padding]
        padded_pic = np.pad(pic, [(self.v_padding, self.v_padding), (self.h_padding, self.h_padding)], mode = self.pad_mode)
        return padded_pic

    def receive(self, movie: list) -> list:
        return [self.apply(frame) for frame in movie]

    def stream(self, movie, buffer_size: int = 1):
        buffer = []
        for frame in movie:
            buffer.append(self.apply(frame))
            if len(buffer) > buffer_size:
                buffer.pop(0)
            yield buffer
