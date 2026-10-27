"""Inference for the feed-forward JAX frame-skip policy shipped in this repo."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from run_rl_policies.environment import ROOT

DEFAULT_MODEL = ROOT / "interactive_gym/server/static/assets/overcooked/models/frame_skip_cramped_room_00.onnx"


class ONNXPolicy:
    def __init__(self, model_path=DEFAULT_MODEL):
        self.model_path = Path(model_path).resolve()
        options = ort.SessionOptions()
        # Small single-observation batches benefit from avoiding thread-pool overhead.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = {item.name: item for item in self.session.get_outputs()}
        if (len(inputs) != 1 or inputs[0].type != "tensor(float)"
                or len(inputs[0].shape) != 2 or "logits" not in outputs):
            raise ValueError("Expected a feed-forward float32 observation input and 'logits' output.")
        self.input = inputs[0]
        self.output = outputs["logits"]
        if len(self.output.shape) != 2 or self.output.type != "tensor(float)":
            raise ValueError("Expected float32 logits with shape [batch, actions].")

    def logits(self, observation):
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        width = self.input.shape[1]
        if isinstance(width, int) and obs.shape[1] != width:
            raise ValueError(f"Model expects {width} features, received {obs.shape[1]}.")
        if not np.isfinite(obs).all():
            raise ValueError("Observation contains non-finite values.")
        logits = self.session.run([self.output.name], {self.input.name: obs})[0][0]
        if not np.isfinite(logits).all():
            raise ValueError("Policy returned non-finite logits.")
        return logits

    def action(self, observation, rng):
        logits = self.logits(observation).astype(np.float64)
        weights = np.exp(logits - np.max(logits))
        return int(rng.choice(len(weights), p=weights / weights.sum()))

    def metadata(self):
        return {
            "inputs": [{"name": item.name, "shape": item.shape, "type": item.type}
                       for item in self.session.get_inputs()],
            "outputs": [{"name": item.name, "shape": item.shape, "type": item.type}
                        for item in self.session.get_outputs()],
            "providers": self.session.get_providers(),
        }
