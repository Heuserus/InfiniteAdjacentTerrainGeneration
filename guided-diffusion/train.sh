MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 2 --save_interval 2000 --use_checkpoint true --num_head_channels 4 --attention_resolutions "32,16,8" --resblock_updown true --learn_sigma true --resume_checkpoint checkpoints/model044000.pt"

TRAINNew_FLAGS="--lr 1e-4 --batch_size 2 --save_interval 2000 --use_checkpoint true --num_head_channels 4 --attention_resolutions "32,16,8" --resblock_updown true --learn_sigma true"
python scripts/image_train.py --data_dir ../Batch256 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS