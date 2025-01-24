import librosa
import numpy as np
import onnxruntime as ort

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
    
    return magnitude[None, ...].astype(np.float32)


class SileroOrt:
    def __init__(self, path: str):
        super().__init__()

        self.batch_size = 1
        sr = 16000
        self.hidden_size = 128
        self.context_size = 64 if sr == 16000 else 32
        self.context = np.zeros((self.batch_size, self.context_size), dtype=np.float32)
        self.state = np.zeros((2, self.batch_size, self.hidden_size), dtype=np.float32)
        self.num_samples = 512 if sr == 16000 else 256

        self.model = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def reset_states(self):
        self.context = np.zeros((self.batch_size, self.context_size), dtype=np.float32)
        self.state = np.zeros((2, self.batch_size, self.hidden_size), dtype=np.float32)

    def __call__(self, x):
        data = np.concatenate([self.context, x], axis=1)
        input_feed = {
            "data": stft_magnitude(data),
            "state": self.state
        }

        output, self.state = self.model.run(None, input_feed=input_feed)
        self.context = x[..., -self.context_size:]
        return output
    
    def audio_forward(self, x):
        outs = []
        self.reset_states()
        num_samples = self.num_samples

        if x.shape[1] % num_samples:
            pad_num = num_samples - (x.shape[1] % num_samples)
            x = np.pad(x, ((0, 0), (0, pad_num)), 'constant', value=0.0)

        for i in range(0, x.shape[1], num_samples):
            wavs_batch = x[:, i:i+num_samples]
            out_chunk = self.__call__(wavs_batch)
            outs.append(out_chunk)

        stacked = np.concatenate(outs, axis=1)
        return stacked