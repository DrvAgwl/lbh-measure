import numpy as np


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
