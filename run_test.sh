#!/usr/bin/env bash
set -e

# ----------------------------------
# Environment
# ----------------------------------
VENV_PATH=".venv"
#export CUDA_VISIBLE_DEVICES=0

# ----------------------------------
# Test configuration
# ----------------------------------
RUN_NAME="dj8rhkwg"
CHECKPOINT="different-sweep-7_best_model_4_0.7806.pt"
#TEST_SET="comparisons_df_with_synthetic_berlin.pickle"
TEST_SET="build_datasets/datasets/comparisons_tests.pkl"
DATASET_DIR="images/printart/subjectivesafety_images"
CITIES="berlin,paris,munich,barcelona,london_uk_collideoscope,london_uk_gov" 
PYTHON_SCRIPT="test.py"

# ----------------------------------
# Activate virtual environment
# ----------------------------------
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: virtualenv not found at $VENV_PATH"
    exit 1
fi

source "$VENV_PATH/bin/activate"

# ----------------------------------
# Run test
# ----------------------------------
python "$PYTHON_SCRIPT" \
  --comparisons "$TEST_SET" \
  --dataset "$DATASET_DIR" \
  --cities "$CITIES" \
  --wandb_run_id "$RUN_NAME" \
  --checkpoint "$CHECKPOINT" \
  --cuda \
  --cuda_id 0

echo "Test finished: $RUN_NAME ($TEST_SET)"


#python "$PYTHON_SCRIPT" \
#    --comparisons "$TEST_SET" \
#    --dataset images/printart/subjectivesafety_images \
#    --cities berlin \
#    --cnn_pool "flatten" \
#    --backbone "vgg" \
#    --checkpoint "vgg_syn+ber.pt" \
#    --model rsscnn \
#    --gaze "off" \
#    --ties \
#    --cuda \
#    --cuda_id 0

#python "$PYTHON_SCRIPT" \
#    --comparisons "$TEST_SET" \
#    --dataset images/printart/subjectivesafety_images \
#    --cities "paris, barcelona, munich" \
#    --wandb_run_id "$RUN_NAME" \
#    --checkpoint "$CHECKPOINT" \
#    --cuda \
#    --cuda_id 0


#BEST NON GAZE:
#RUN_NAME="mbg8xvwg"
#CHECKPOINT="lemon-sweep-1_best_model_10_0.7847.pt"
#Best gaze:
#RUN_NAME="dw1th1vv"
#CHECKPOINT="confused-sweep-2_best_model_8_0.7899.pt"

#RUN_NAME="rwoa07zg"
#CHECKPOINT="divine-sweep-1_best_model_6_0.8021.pt"