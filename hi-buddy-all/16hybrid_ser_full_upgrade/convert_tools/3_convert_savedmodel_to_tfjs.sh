#!/bin/bash
# 3_convert_savedmodel_to_tfjs.sh
SAVED=$1; OUTDIR=$2
if [ -z "$SAVED" ] || [ -z "$OUTDIR" ]; then echo "Usage: $0 /path/to/saved_model /path/to/out_tfjs"; exit 1; fi
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_layers_model --signature_name=serving_default "$SAVED" "$OUTDIR"
echo "Converted TF SavedModel -> TFJS at $OUTDIR"
