import torch
import numpy as np




def prepare_input(pcd_points, pcd_colors, input_type='tensor', device=None):
    input_shape = (1, pcd_points.shape[0], 9)
    if input_type == 'tensor':
        input_tensor = torch.zeros(input_shape).to(device)
        input_tensor[0, :, 0:3] = torch.tensor(pcd_points)
        try:
            input_tensor[0, :, 3:6] = torch.tensor(pcd_colors)
        except RuntimeError:
            print("Could not find colors")
            pass
        return input_tensor
    elif input_type == 'np':
        input_array = np.zeros(input_shape)
        input_array[0, :, 0:3] = pcd_points
        try:
            input_array[0, :, 3:6] = pcd_colors
        except RuntimeError:
            print("Could not find colors")
            pass
        return input_array
    else:
        raise InputTypeNotFoundException(f'The give input type {input_type} is not found. Only allowed - tensor, np')


def invoke_torch_model(model, input_tensor):
    input_tensor = input_tensor.permute(0, 2, 1)
    return model(input_tensor)


def invoke_onnx_model(ort_session, input_array: np.array):
    # ort_session = ort.InferenceSession('/tmp/test.onnx')
    input_name = ort_session.get_inputs()[0].name
    input_array = np.transpose(input_array, (0, 2, 1))
    ort_inputs = {input_name: input_array.astype(np.float32)}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs


class InputTypeNotFoundException(Exception):
    pass
