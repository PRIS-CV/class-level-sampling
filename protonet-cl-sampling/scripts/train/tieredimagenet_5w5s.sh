python train.py -batch 256 -dataset tieredimagenet -gpu 0,1 -extra_dir your_run -temperature_attn 5.0 -lamb 0.25 -shot 5 -milestones 40 50 -max_epoch 60 -train_sampling cl_sampling

