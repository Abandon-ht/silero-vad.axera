from silero import *
import librosa
import numpy as np
from SileroOrt import SileroOrt

def stft_magnitude(input_data, n_fft=256, hop_length=128):
    """
    计算输入音频数据的STFT幅度谱。
    
    参数:
        input_data (np.ndarray): 输入的音频时间序列数据。
        n_fft (int): 每个STFT帧的FFT窗口大小，默认为256。
        hop_length (int): 帧之间的样本数，默认为128。
        
    返回:
        np.ndarray: STFT的幅度谱。
    """
    # 计算需要填充的数量
    pad_amount = (n_fft - hop_length) // 2
    
    # 使用反射填充
    padded_input = np.pad(input_data[0], (0, pad_amount), mode='reflect')

    # 使用librosa计算STFT
    D = librosa.stft(padded_input, n_fft=n_fft, hop_length=hop_length, center=False)
    
    # 获取幅度谱
    magnitude = np.abs(D)
    
    return magnitude[None, ...]

# 示例用法：
# audio_data, sr = librosa.load('your_audio_file.wav')
# magnitude_spectrum = stft_magnitude(audio_data)


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

