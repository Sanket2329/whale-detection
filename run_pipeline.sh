#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================================="
echo "      Whale Vocalization SED Generalization Pipeline     "
echo "=========================================================="

export DYLD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/torch/lib
export PYTHONPATH=.

echo "Step 1: Training CRNN on Casey (2014) and validating on Greenwich (2015)"
echo "----------------------------------------------------------"
python3 -u src/train.py \
    --train_dir data/casey2014 \
    --train_name casey2014 \
    --val_dir data/Greenwich64S2015 \
    --val_name Greenwich64S2015 \
    --epochs 10 \
    --batch_size 128 \
    --lr 0.001 \
    --save_path models/best_model.pth \
    --subsample_train 0.1 \
    --subsample_val 0.2 \
    --neg_multiplier 1.0 \
    --loss_type focal \
    --mixup_alpha 0.2 \
    --spec_augment

echo ""
echo "Step 2: Running Sliding Window Inference on Greenwich (2015)"
echo "----------------------------------------------------------"
python3 -u src/infer.py \
    --model_path models/best_model.pth \
    --wav_path data/Greenwich64S2015/wav \
    --output_dir reports/predictions/Greenwich64S2015 \
    --threshold 0.5

echo ""
echo "Step 3: Evaluating Performance & Generalization"
echo "----------------------------------------------------------"
python3 -u src/evaluate.py \
    --gt_dir data/Greenwich64S2015 \
    --gt_name Greenwich64S2015 \
    --pred_dir reports/predictions/Greenwich64S2015 \
    --iou_thresh 0.1

echo ""
echo "Step 4: Exporting to ONNX and Benchmarking Inference Speed"
echo "----------------------------------------------------------"
python3 -m src.export_onnx
python3 -m src.benchmark_onnx

echo ""
echo "=========================================================="
echo "               Pipeline Execution Finished                "
echo "=========================================================="
