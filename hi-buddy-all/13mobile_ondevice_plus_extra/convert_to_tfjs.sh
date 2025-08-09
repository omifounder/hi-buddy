#!/bin/bash
# Convert TensorFlow SavedModel to TFJS Layers format
# Prereqs:
# pip install tensorflowjs
# Usage: ./convert_to_tfjs.sh /path/to/saved_model /path/to/output_tfjs
set -e
SAVED_MODEL=$1
OUT_DIR=$2
if [ -z "$SAVED_MODEL" ] || [ -z "$OUT_DIR" ]; then
  echo "Usage: $0 /path/to/saved_model /path/to/output_tfjs"
  exit 1
fi
rm -rf "$OUT_DIR"
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_layers_model "$SAVED_MODEL" "$OUT_DIR"
echo "Converted to TFJS at $OUT_DIR"
