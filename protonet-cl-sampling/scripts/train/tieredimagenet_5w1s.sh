python train.py -batch 256 -dataset tieredimagenet -gpu 2,3 -extra_dir your_run -temperature_attn 5.0 -lamb 0.25 -milestones 60 80 -max_epoch 100 -train_sampling cl_sampling
