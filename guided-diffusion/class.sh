TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 2 --lr 3e-4 --save_interval 2000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 64 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"

mpiexec -n N python scripts/classifier_train.py --data_dir datasets/hmaps_train_8bit $TRAIN_FLAGS $CLASSIFIER_FLAGS