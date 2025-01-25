from silero import *
import numpy as np
from SileroOrt import SileroOrt
from utils_vad import stft_magnitude


if __name__ == "__main__":
    jit_model = torch.jit.load("./silero_vad.jit")
    jit_model.eval()
    state_dict = jit_model.state_dict()
    state_dict['_model.stft.forward_basis_buffer.weight'] = state_dict['_model.stft.forward_basis_buffer']

    ort_model = SileroOrt("./silero_vad.onnx")

    batch_size = 1
    sr = 16000
    hidden_size = 128
    context_size = 64 if sr == 16000 else 32
    context = torch.zeros((batch_size, context_size))
    state = np.zeros((2, batch_size, hidden_size), dtype=np.float32)
    num_samples = 512 if sr == 16000 else 256

    # Compare STFT
    stft_module = STFT()
    stft_module.eval()
    stft_module.forward_basis_buffer.weight.data = state_dict['_model.stft.forward_basis_buffer.weight']

    with torch.no_grad():
        input_tensor = torch.randn(1, num_samples)
        input_tensor = torch.cat([context, input_tensor], dim=1)

        stft_gt = stft_module(input_tensor)
        stft_gt = stft_gt.numpy()

        stft_np = stft_magnitude(input_tensor.numpy())

        np.testing.assert_allclose(stft_np, stft_gt, atol=1e-5)

    # Compare PyTorch and ONNX
    with torch.no_grad():
        for i in range(10):
            # Perform forward pass
            input_tensor = torch.randn(1, num_samples)  # Sample input (batch_size=10, feature_dim=256)

            jit_output = jit_model(input_tensor, sr)

            output = ort_model(input_tensor.float().numpy())

            np.testing.assert_allclose(output, jit_output.numpy(), atol=1e-5)

            print(f"Compare {i} success")

