#
#============================#
# Setup
#============================#
# cd /project/community/$(whoami)/FLARE-dev.py
source .venv/bin/activate

#======================================================================#
MODEL=set_transformer
#======================================================================#

DATASET=lpbf
CUDA_VISIBLE_DEVICES=2 python -m pdebench \
  --dataset.dataset "${DATASET}" --run.train true --model.model ${MODEL} --run.exp_name model_${MODEL}_${DATASET} --use_puri2025flare_config true &

DATASET=drivaerml_40k
CUDA_VISIBLE_DEVICES=3 python -m pdebench \
  --dataset.dataset "${DATASET}" --run.train true --model.model ${MODEL} --run.exp_name model_${MODEL}_${DATASET} --use_puri2025flare_config true &

# #======================================================================#
# MODEL_TYPE=mambano
# #======================================================================#

# DATASET=darcy
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=gpu --master_port 29510 -m pdebench --dataset.dataset ${DATASET} --run.train true --use_puri2025flare_config true \
#     --model.model ${MODEL_TYPE} --run.exp_name model_${MODEL_TYPE}_${DATASET} &

# wait

# DATASET=airfoil_steady
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=gpu --master_port 29511 -m pdebench --dataset.dataset ${DATASET} --run.train true --use_puri2025flare_config true \
#     --model.model ${MODEL_TYPE} --run.exp_name model_${MODEL_TYPE}_${DATASET} &

# wait

# DATASET=pipe
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=gpu --master_port 29512 -m pdebench --dataset.dataset ${DATASET} --run.train true --use_puri2025flare_config true \
#     --model.model ${MODEL_TYPE} --run.exp_name model_${MODEL_TYPE}_${DATASET} &

# exit

#======================================================================#
# MODEL_TYPE=lamo
#======================================================================#

# DATASET=elasticity
# CUDA_VISIBLE_DEVICES=0 python -m pdebench --dataset.dataset ${DATASET} --run.train true --use_puri2025flare_config true \
#     --model.model ${MODEL_TYPE} --run.exp_name model_${MODEL_TYPE}_${DATASET} &

# DATASET=darcy
# CUDA_VISIBLE_DEVICES=1 python -m pdebench --dataset.dataset ${DATASET} --run.train true --use_puri2025flare_config true \
#     --model.model ${MODEL_TYPE} --run.exp_name model_${MODEL_TYPE}_${DATASET} &

# DATASET=airfoil_steady
# CUDA_VISIBLE_DEVICES=2 python -m pdebench --dataset.dataset ${DATASET} --run.train true --use_puri2025flare_config true \
#     --model.model ${MODEL_TYPE} --run.exp_name model_${MODEL_TYPE}_${DATASET} &

# DATASET=pipe
# CUDA_VISIBLE_DEVICES=2 python -m pdebench --dataset.dataset ${DATASET} --run.train true --use_puri2025flare_config true \
#     --model.model ${MODEL_TYPE} --model.conv2d true --run.exp_name model_${MODEL_TYPE}_conv2d_${DATASET} &

# DATASET=pipe
# CUDA_VISIBLE_DEVICES=3 python -m pdebench --dataset.dataset ${DATASET} --run.train true --use_puri2025flare_config true \
#     --model.model ${MODEL_TYPE} --model.conv2d false --run.exp_name model_${MODEL_TYPE}_${DATASET} &

# DATASET=drivaerml_40k
# CUDA_VISIBLE_DEVICES=2 python -m pdebench --dataset.dataset ${DATASET} --run.train true --use_puri2025flare_config true \
#     --model.model ${MODEL_TYPE} --run.exp_name model_${MODEL_TYPE}_${DATASET} &

# DATASET=lpbf
# CUDA_VISIBLE_DEVICES=3 python -m pdebench --dataset.dataset ${DATASET} --run.train true --use_puri2025flare_config true \
#     --model.model ${MODEL_TYPE} --run.exp_name model_${MODEL_TYPE}_${DATASET} &

#======================================================================#
# MODEL_TYPE=transolver++
#======================================================================#

# DATASET=elasticity
# CUDA_VISIBLE_DEVICES=0 python -m pdebench --dataset.dataset ${DATASET} --run.train true --use_puri2025flare_config true \
#     --model.model ${MODEL_TYPE} --run.exp_name model_${MODEL_TYPE}_${DATASET} &

# DATASET=darcy
# CUDA_VISIBLE_DEVICES=1 python -m pdebench --dataset.dataset ${DATASET} --run.train true --use_puri2025flare_config true \
#     --model.model ${MODEL_TYPE} --run.exp_name model_${MODEL_TYPE}_${DATASET} &

# DATASET=airfoil_steady
# CUDA_VISIBLE_DEVICES=2 python -m pdebench --dataset.dataset ${DATASET} --run.train true --use_puri2025flare_config true \
#     --model.model ${MODEL_TYPE} --run.exp_name model_${MODEL_TYPE}_${DATASET} &

# DATASET=pipe
# CUDA_VISIBLE_DEVICES=3 python -m pdebench --dataset.dataset ${DATASET} --run.train true --use_puri2025flare_config true \
#     --model.model ${MODEL_TYPE} --run.exp_name model_${MODEL_TYPE}_${DATASET} &

# DATASET=drivaerml_40k
# CUDA_VISIBLE_DEVICES=0 python -m pdebench --dataset.dataset ${DATASET} --run.train true --use_puri2025flare_config true \
#     --model.model ${MODEL_TYPE} --run.exp_name model_${MODEL_TYPE}_${DATASET} &

# DATASET=lpbf
# CUDA_VISIBLE_DEVICES=0 python -m pdebench --dataset.dataset ${DATASET} --run.train true --use_puri2025flare_config true \
#     --model.model ${MODEL_TYPE} --run.exp_name model_${MODEL_TYPE}_${DATASET} &

#======================================================================#
#
