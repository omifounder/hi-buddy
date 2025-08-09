# 0_download_and_export_to_onnx.py
import argparse, os, torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_id', required=True)
    p.add_argument('--out_dir', default='./exported')
    p.add_argument('--opset', type=int, default=14)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print('Loading model:', args.model_id)
    model = AutoModelForAudioClassification.from_pretrained(args.model_id)
    model.eval()
    dummy_audio = torch.randn(1,16000)
    input_names = ['input_values']
    output_names = ['logits']
    onnx_path = os.path.join(args.out_dir, 'model.onnx')
    torch.onnx.export(model, args=(dummy_audio,), f=onnx_path, input_names=input_names, output_names=output_names, opset_version=args.opset, dynamic_axes={'input_values':{1:'sequence'}, 'logits':{1:'classes'}}, do_constant_folding=True)
    print('Exported ONNX to', onnx_path)

if __name__ == '__main__':
    main()
