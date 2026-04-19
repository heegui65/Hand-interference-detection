# infer.py
import onnxruntime as ort
import cv2
import numpy as np
from config import TARGET_SIZE

class HandDetector:
    def __init__(self, model_path="best.onnx"):
        # Raspberry Pi ONNX Runtime config: disable GPU, use CPU optimization
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4  # Use 4 threads for Raspberry Pi 4B/5
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(model_path, sess_options=sess_options)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        """Preprocess image to fit model input (640x640)."""
        img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))
        img = img / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    def infer(self, img):
        """Run inference and return model outputs."""
        input_tensor = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return outputs