import cv2
import onnxruntime as rt
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

work_dir = os.path.dirname(os.path.realpath(__file__))

class FaceNet():

    def __init__(self, model_name, rescale, input_shape, backend='onnx'):
        self.rescale = rescale
        self.input_shape = input_shape
        if backend == 'onnx' or backend == 'cuda':
            provider = ['CUDAExecutionProvider' if backend=='cuda' else 'CPUExecutionProvider']
            self.model = rt.InferenceSession(os.path.join(work_dir, model_name), providers=provider)
            self.input_name = self.model.get_inputs()[0].name
        elif backend == 'tf':
            self.model = tf.keras.models.load_model(model_name)

    # normalize images
    def prewhiten(self, image_array):
        axis = (0, 1, 2)
        size = image_array.size
        mean = np.mean(image_array, axis=axis, keepdims=True)
        std = np.std(image_array, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0/np.sqrt(size))
        y = (image_array - mean) / std_adj
        return y

    def generate_embeddings(self, image):
        # avoid changing the image passed
        image = image.copy()
        image = cv2.resize(image, self.input_shape)
        image = image.reshape(1, *self.input_shape, 3)
        image = self.prewhiten(image)
        outputs = []
        if isinstance(self.model,  tf.keras.Model):
            outputs = self.model.predict(image)
        elif isinstance(self.model, rt.capi.onnxruntime_inference_collection.InferenceSession):
            image = image.astype(np.float32)
            outputs = self.model.run([], {self.input_name : image})
        return outputs[0]
