import os
import tempfile
import requests
import numpy as np
import onnxruntime as ort
from onnxruntime import OrtDevice
from onnxruntime.quantization import quantize_dynamic, QuantType
from PIL import Image
from torchvision import transforms

class CachedWoundClassifier:
    MODEL_URL = "https://huggingface.co/PogusTheWhisper/Surgicare-ALB-fold4-stage3/resolve/main/topdown_model_fold4_stage3_opset_20.onnx"
    CACHE_DIR = tempfile.gettempdir()
    MODEL_PATH = os.path.join(CACHE_DIR, "model_fp32.onnx")
    QUANT_PATH = os.path.join(CACHE_DIR, "model_fp16.onnx")
    CLASS_LABELS = {
        0: 'Abrasions',
        1: 'Bruises',
        2: 'Burns',
        3: 'Cut',
        4: 'Normal'
    }

    def __init__(self, use_fp16=True, device=None):
        providers = ort.get_available_providers()
        use_cuda = "CUDAExecutionProvider" in providers
        self.device = device or ("cuda" if use_cuda else "cpu")
        if not os.path.exists(self.MODEL_PATH):
            r = requests.get(self.MODEL_URL, stream=True)
            r.raise_for_status()
            with open(self.MODEL_PATH, "wb") as f:
                for c in r.iter_content(1 << 20):
                    f.write(c)
        model_path = self.MODEL_PATH
        if use_fp16 and use_cuda:
            if not os.path.exists(self.QUANT_PATH):
                quantize_dynamic(self.MODEL_PATH, self.QUANT_PATH, weight_type=QuantType.QFloat16)
            model_path = self.QUANT_PATH
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.intra_op_num_threads = os.cpu_count()
        so.inter_op_num_threads = 1
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True
        so.log_severity_level = 3
        if self.device == "cuda":
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [
                {"trt_max_workspace_size": "1073741824", "trt_fp16_enable": "1"},
                {"device_id": 0, "cudnn_conv_algo_search": "EXHAUSTIVE", "arena_extend_strategy": "kNextPowerOfTwo"},
                {}
            ]
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]
        self.model = ort.InferenceSession(model_path, sess_options=so, providers=providers, provider_options=provider_options)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        if self.device == "cuda":
            shape = [1] + [dim if dim is not None else 1 for dim in self.model.get_inputs()[0].shape[1:]]
            self._input_buffer = ort.OrtValue.ortvalue_from_numpy(
                np.zeros(shape, dtype=np.float32),
                OrtDevice("cuda", 0)
            )
            self.iobinding = self.model.io_binding()
            self.iobinding.bind_input(
                name=self.input_name,
                device_type="cuda",
                device_id=0,
                element_type=np.float32,
                shape=shape,
                buffer_ptr=self._input_buffer.data_ptr()
            )
        else:
            self.iobinding = None
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).numpy()
        if self.iobinding:
            np.copyto(self._input_buffer.numpy(), tensor)
            self.iobinding.bind_output(
                name=self.output_name,
                device_type="cuda",
                device_id=0
            )
            self.model.run_with_iobinding(self.iobinding)
            out = self.iobinding.get_outputs()[0].numpy()
        else:
            out = self.model.run([self.output_name], {self.input_name: tensor})[0]
        e = np.exp(out - out.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        idx = int(probs.argmax())
        label = self.CLASS_LABELS.get(idx, "Unknown")
        return label, probs
