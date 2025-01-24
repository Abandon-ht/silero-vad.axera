# silero-vad.axera
Silero VAD implementation on Axera platforms

Thanks to https://github.com/lovemefan/Silero-vad-pytorch/tree/main, a reverse engineering implementation of https://github.com/snakers4/silero-vad


## 导出ONNX
```
python export_onnx.py
```
生成silero_vad.onnx

## 对比ONNX和PyTorch
```
python compare.py
```