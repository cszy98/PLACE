CUDA_VISIBLE_DEVICE=0 python inference.py --outdir output/ade20k \
                                          --config configs/stable-diffusion/PLACE.yaml \
                                          --ckpt ckpt/ade20k_best.ckpt \
                                          --dataset ADE20K \
                                          --data_root path_to_ADE20K
