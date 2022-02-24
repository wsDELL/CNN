import numpy as np
from numba import jit
from pathlib import Path

# def pool_forward():
import MiniFramework


def img2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    img = input_data
    if pad > 0:
        img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype='float32')

    for i in range(filter_h):
        i_max = i + stride * out_h
        for j in range(filter_w):
            j_max = j + stride * out_w
            col[:, :, i, j, :, :] = img[:, :, i:i_max:stride, j:j_max:stride]
    col = np.transpose(col, axes=(0, 4, 5, 1, 2, 3)).reshape(N * out_h * out_w, -1)
    return col


def col2img(col, input_shape, filter_h, filter_w, stride, pad, out_h, out_w):
    N, C, H, W = input_shape
    tmp1 = col.reshape(N, out_h, out_w, C, filter_h, filter_w).astype('float32')
    tmp2 = np.transpose(tmp1, axes=(0, 3, 4, 5, 1, 2))
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for i in range(filter_h):
        i_max = i + stride * out_h
        for j in range(filter_w):
            j_max = j + stride * out_w
            img[:, :, i:i_max:stride, j:j_max:stride] += tmp2[:, :, i, j, :, :]
    return img[:, :, pad:H + pad, pad:W + pad]


def calculate_output_size(input_h, input_w, filter_h, filter_w, padding, stride=1):
    output_h = (input_h - filter_h + 2 * padding) // stride + 1
    output_w = (input_w - filter_w + 2 * padding) // stride + 1
    result = (output_h, output_w)
    return result


def jit_cov2d(input_v: np.ndarray, weights: np.ndarray, bias, out_w, out_h, stride=1):
    assert (input_v.ndim == 4)
    assert (input_v.shape[1] == weights.shape[1])
    batch_size = input_v.shape[0]
    input_channel = input_v.shape[1]
    output_channel = weights.shape[0]
    filter_height = weights.shape[1]
    filter_width = weights.shape[2]
    rs = np.zeros((batch_size, output_channel, out_h, out_w))

    for bs in range(batch_size):
        for oc in range(output_channel):
            rs[bs, oc] += bias[oc]
            for ic in range(input_channel):
                for i in range(out_h):
                    for j in range(out_w):
                        ii = i * stride
                        jj = j * stride
                        for fh in range(filter_height):
                            for fw in range(filter_width):
                                rs[bs, oc, i, j] += input_v[bs, ic, ii + fh, jj + fw] * weights[oc, ic, fh, fw]

    return rs


def jit_conv_2d(input_array, kernal, bias, output_array):
    assert (input_array.ndim == 2)
    assert (output_array.ndim == 2)
    assert (kernal.ndim == 2)

    output_height = output_array.shape[0]
    output_width = output_array.shape[1]
    kernal_height = kernal.shape[0]
    kernal_width = kernal.shape[1]

    for i in range(output_height):
        i_start = i
        i_end = i_start + kernal_height
        for j in range(output_width):
            j_start = j
            j_end = j_start + kernal_width
            target_array = input_array[i_start:i_end, j_start:j_end]
            output_array[i, j] = np.sum(target_array * kernal) + bias


def expand_delta_map(delta_in, batch_size, input_c, input_h, input_w, output_h, output_w, filter_h, filter_w, padding,
                     stride):
    assert (delta_in.ndim == 4)
    expand_h = 0
    expand_w = 0
    if stride == 1:
        dZ_stride_1 = delta_in
        expand_h = delta_in.shape[2]
        expand_w = delta_in.shape[3]
    else:
        (expand_h, expand_w) = calculate_output_size(input_h, input_w, filter_h, filter_w, padding, 1)
        dZ_stride_1 = np.zeros((batch_size, input_c, expand_h, expand_w))
        for bs in range(batch_size):
            for ic in range(input_c):
                for i in range(output_h):
                    for j in range(output_w):
                        ii = i * stride
                        jj = j * stride
                        dZ_stride_1[bs, ic, ii, jj] = delta_in[bs, ic, i, j]

    return dZ_stride_1


def calculate_padding_size(input_h, input_w, filter_h, filter_w, output_h, output_w, stride=1):
    pad_h = ((output_h - 1) * stride - input_h + filter_h) // 2
    pad_w = ((output_w - 1) * stride - input_w + filter_w) // 2
    return (pad_h, pad_w)


def calcalate_weights_grad(x, dz, batch_size, output_c, input_c, filter_h, filter_w, dW, dB):
    for bs in range(batch_size):
        for oc in range(output_c):  # == kernal count
            for ic in range(input_c):  # == filter count
                w_grad = np.zeros((filter_h, filter_w)).astype(np.float32)
                # w_grad = np.zeros((filter_h, filter_w))
                jit_conv_2d(x[bs, ic], dz[bs, oc], 0, w_grad)
                dW[oc, ic] += w_grad
            # end ic
            dB[oc] += dz[bs, oc].sum()
        # end oc
    # end bs
    return (dW, dB)


def calculate_delta_out(dz, rot_weights, batch_size, num_input_channel, num_output_channel, input_height, input_width,
                        delta_out):
    for bs in range(batch_size):
        for oc in range(num_output_channel):    # == kernal count
            delta_per_input = np.zeros((input_height, input_width)).astype(np.float32)
            #delta_per_input = np.zeros((input_height, input_width))
            for ic in range(num_input_channel): # == filter count
                jit_conv_2d(dz[bs,oc], rot_weights[oc,ic], 0, delta_per_input)
                delta_out[bs,ic] += delta_per_input
            #END IC
        #end oc
    #end bs
    return delta_out

