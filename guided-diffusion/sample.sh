MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 2 --num_head_channels 4 --attention_resolutions "32,16,8" --resblock_updown true --learn_sigma true"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"

python scripts/image_sample.py --model_path ../models/64Terra.pt $MODEL_FLAGS $DIFFUSION_FLAGS --num_samples 16