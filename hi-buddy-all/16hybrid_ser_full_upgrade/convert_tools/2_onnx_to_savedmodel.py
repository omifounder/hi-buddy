# 2_onnx_to_savedmodel.py
import argparse, os
from onnx_tf.backend import prepare
import onnx

p = argparse.ArgumentParser()
p.add_argument('--onnx', required=True)
p.add_argument('--out', default='./saved_model')
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
onnx_model = onnx.load(args.onnx)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(args.out)
print('SavedModel exported to', args.out)
