
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file ../config/GRPO3.yaml \
--num_processes=3 ../src/grpo_r1/grpo.py \
--config ../config/GRPO_R1_zero_1dot5B_config.yaml \
> ./output/grpo_r1_1dot5B_sampling.log 2>&1
