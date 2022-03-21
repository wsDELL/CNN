import numpy as np


def _calculate_fan_in_and_fan_out(tensor: np.ndarray):
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError("Fan in and Fan out can not be computed for tensor with less than 2 dimensions")
    if dimensions == 2:
        fan_in = tensor.shape[1]
        fan_out = tensor.shape[0]
    else:
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
        receptive_field_size = 1
        if tensor.ndim > 2:
            receptive_field_size = tensor.size / (tensor.shape[0] * tensor.shape[1])
            receptive_field_size = int(receptive_field_size)
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


if __name__ == "__main__":
    x = np.random.rand(64, 3, 3, 3)
    y = _calculate_fan_in_and_fan_out(x)
    print(y)
