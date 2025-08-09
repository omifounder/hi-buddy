# 1_quantize_onnx.py
import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType

parser = argparse.ArgumentParser()
parser.add_argument('--onnx', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--mode', default='dynamic', choices=['dynamic','static'])
args = parser.parse_args()

if args.mode == 'dynamic':
    quantize_dynamic(model_input=args.onnx, model_output=args.out, weight_type=QuantType.QInt8)
    print('Dynamic quantization complete ->', args.out)
else:
    print('Static quantization not implemented here')
