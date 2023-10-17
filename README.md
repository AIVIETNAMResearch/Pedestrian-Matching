Running Script:
```
python3 Retrieval.py --config "configs/finetune/icfg_pedes.yaml" --checkpoint "checkpoint/x2vlm_base_4m.th" --output_dir "output/tmp"
```

Udate Script  - save best r1
```
python3 Retrieval.py --config "configs/finetune/icfg_pedes_ckc.yaml" --checkpoint "checkpoint/x2vlm_base_1b.th" --output_dir "output/Baseline_1B_CKC" --pick_best_r1
```