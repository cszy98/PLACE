CUDA_VISIBLE_DEVICE=0 python inference.py --outdir output/COCO \
                                          --config configs/stable-diffusion/PLACE.yaml \
                                          --ckpt ckpt/coco_best.ckpt  \
                                          --dataset COCO \
                                          --data_root path_to_COCO
