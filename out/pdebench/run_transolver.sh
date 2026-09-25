#
#============================#
# Setup
#============================#
source .venv/bin/activate

#======================================================================#
MODEL=transolver
#======================================================================#

DATASET=elasticity
CUDA_VISIBLE_DEVICES=0 python -m pdebench \
  --dataset.dataset "${DATASET}" --run.train true --model.model ${MODEL} --run.exp_name model_${MODEL}_upstream_${DATASET} --use_puri2025flare_config true &

DATASET=navier_stokes
CUDA_VISIBLE_DEVICES=1 python -m pdebench \
  --dataset.dataset "${DATASET}" --run.train true --model.model ${MODEL} --run.exp_name model_${MODEL}_upstream_${DATASET} --use_puri2025flare_config true &

DATASET=plasticity
CUDA_VISIBLE_DEVICES=2 python -m pdebench \
  --dataset.dataset "${DATASET}" --run.train true --model.model ${MODEL} --run.exp_name model_${MODEL}_upstream_${DATASET} --use_puri2025flare_config true &

# wait

#======================================================================#
#
