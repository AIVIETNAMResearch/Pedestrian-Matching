CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 \
        --nnodes=1 Retrieval.py \
        --config "./configs/finetune/cuhk_pedes.yaml" --checkpoint "./checkpoint/x2vlm_base_1b.th" --output_dir "./output/tmp/" --pick_best_r1 