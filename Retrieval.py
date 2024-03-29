import argparse
import os
import sys
import math

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist


import utils
from utils.checkpointer import Checkpointer
from utils.hdfs_io import hmkdir
from utils.mlm_tool import mlm, TextMaskingGenerator, NounMaskingGenerator

from dataset import create_dataset, create_sampler, create_loader, build_tokenizer
from scheduler import create_scheduler
from optim import create_optimizer
from models.model_retrieval import CFACKCModel

def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    print("DEVICE: ", device)

    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config["mlm"]:
        metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100 
    #scaler = torch.cuda.amp.GradScaler()


    accumulate_steps = int(config.get('accumulate_steps', 1))
    for i, (image1, image2, image_cnn, caption1, caption2, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image1 = image1.to(device, non_blocking=True)
        image2 = image2.to(device, non_blocking=True)
        image_cnn = image_cnn.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        text_input1 = tokenizer(caption1, padding='max_length', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
        text_input2 = tokenizer(caption2, padding='max_length', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
        
        #optimizer.zero_grad()
        if config["mlm"]:
            if config["mask_type"] == "default":
                mask_generator = TextMaskingGenerator(tokenizer, config['mask_prob'], config['max_masks'],
                                                    config['skipgram_prb'], config['skipgram_size'],
                                                    config['mask_whole_word'])  
            elif config["mask_type"] == "noun":
                mask_generator = NounMaskingGenerator(tokenizer, config['mask_prob'], config['max_masks'])
            else:
                raise ValueError("mask_type should be either default or noun")          
                                                
            text_ids_masked, masked_pos, masked_ids = mlm(caption1, text_input1, tokenizer, device, mask_generator, config)
            mlm_input = (text_ids_masked, masked_pos, masked_ids)

            if config["use_id_loss"]:
                loss_itc, loss_itm, loss_id, loss_mlm = model(image1, image2, image_cnn, text_input1.input_ids, text_input1.attention_mask, 
                                    text_input2.input_ids, text_input2.attention_mask, idx=idx, mlm_inputs=mlm_input)
                loss = loss_itc + loss_itm + loss_id + loss_mlm 
            else:
                #text_input_clip = None
                #if config["use_clip_feats"]:
                #    text_input_clip = clip.tokenize(caption1, truncate=True).to(device)

                loss_itc, loss_itm, loss_mlm = model(image1, image2, image_cnn, text_input1.input_ids, text_input1.attention_mask, 
                                        text_input2.input_ids, text_input2.attention_mask, idx=idx, mlm_inputs=mlm_input) 
                                        #text_input_clip=text_input_clip, image_cnn=image_cnn)
                loss = loss_itc + loss_itm + loss_mlm
        else:
            loss_itc, loss_itm = model(image1, image2, image_cnn, text_input1.input_ids, text_input1.attention_mask, 
                                    text_input2.input_ids, text_input2.attention_mask, idx=idx)

            loss = loss_itc + loss_itm
        if accumulate_steps > 1:
            loss = loss / accumulate_steps
        # backward
        loss.backward()

        if (i+1) % accumulate_steps == 0:
            #update
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_itc=loss_itc.item())
        if config["mlm"]:
            metric_logger.update(loss_mlm=loss_mlm.item())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # evaluate
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print('Computing features for evaluation...')
    start_time = time.time()
    # extract text features
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
        text_feat = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
        text_embed = model.get_features(text_embeds=text_feat)

        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed.detach().cpu())
        text_feats.append(text_feat.detach().cpu())
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    # extract image features
    image_feats = []
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)

        image_feat, _ = model.get_vision_embeds(image)
        image_embed = model.get_features(image_embeds=image_feat)
        image_feats.append(image_feat.detach().cpu())
        image_embeds.append(image_embed.detach().cpu())

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    # compute the feature similarity score for all image-text pairs
    sims_matrix = text_embeds @ image_embeds.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)
    # take the top-k candidates and calculate their ITM score sitm for ranking
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.get_cross_embeds(image_embeds=encoder_output.to(device), image_atts=encoder_att.to(device),
                                         text_embeds=text_feats[start + i].repeat(config['k_test'], 1, 1).to(device),
                                         text_atts=text_atts[start + i].repeat(config['k_test'], 1).to(device))

        score = model.itm_head(output[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score
    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))
    return score_matrix_t2i.cpu()

@torch.no_grad()
def itm_eval(scores_t2i, img2person, txt2person, eval_mAP):
    img2person = torch.tensor(img2person)
    txt2person = torch.tensor(txt2person)
    index = torch.argsort(scores_t2i, dim=-1, descending=True)
    pred_person = img2person[index]
    matches = (txt2person.view(-1, 1).eq(pred_person)).long()

    def acc_k(matches, k=1):
        matches_k = matches[:, :k].sum(dim=-1)
        matches_k = torch.sum((matches_k > 0))
        return 100.0 * matches_k / matches.size(0)

    # Compute metrics
    ir1 = acc_k(matches, k=1).item()
    ir5 = acc_k(matches, k=5).item()
    ir10 = acc_k(matches, k=10).item()
    ir_mean = (ir1 + ir5 + ir10) / 3

    if eval_mAP:
        real_num = matches.sum(dim=-1)
        tmp_cmc = matches.cumsum(dim=-1).float()
        order = torch.arange(start=1, end=matches.size(1) + 1, dtype=torch.long)
        tmp_cmc /= order
        tmp_cmc *= matches
        AP = tmp_cmc.sum(dim=-1) / real_num
        mAP = AP.mean() * 100.0
        eval_result = {'r1': ir1,
                       'r5': ir5,
                       'r10': ir10,
                       'r_mean': ir_mean,
                       'mAP': mAP.item()
                       }
    else:
        eval_result = {'r1': ir1,
                       'r5': ir5,
                       'r10': ir10,
                       'r_mean': ir_mean,
                       }
    return eval_result

def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if args.epoch > 0:
        config['schedular']['epochs'] = args.epoch
        print(f"### set epochs to: {args.epoch}", flush=True)

    if args.bs > 0:
        config['batch_size_train'] = args.bs // world_size

    if args.k_test > 0:
        config['k_test'] = args.k_test
        print(f"### set k_test to: {args.k_test}", flush=True)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print(f"Creating model", flush=True)
    model = CFACKCModel(config=config)

    model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate, use_mlm_loss=config["mlm"])
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    tokenizer = build_tokenizer(config['text_tokenizer'])

    print("Creating retrieval dataset", flush=True)

    if config.get('is_video', False):
        train_dataset, test_dataset = create_dataset('re_video', config, args.evaluate)
        val_dataset = test_dataset
    else:
        train_dataset, val_dataset, test_dataset = create_dataset('ps', config, args.evaluate)

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    if args.evaluate:
        print("Start evaluating", flush=True)
        if utils.is_main_process():
            print(f"### data {len(test_dataset)}")

        test_loader = create_loader([test_dataset], [None],
                                    batch_size=[config['batch_size_test']],
                                    num_workers=[4],
                                    is_trains=[False],
                                    collate_fns=[None])[0]

        # score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
        score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

        if utils.is_main_process():
            # val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)
            # print(val_result)
            test_result = itm_eval(score_test_t2i, test_loader.dataset.img2person, test_loader.dataset.txt2person, eval_mAP=True)
            print(test_result)

        dist.barrier()

    else:
        print("Start training", flush=True)

        train_dataset_size = len(train_dataset)

        if utils.is_main_process():
            print(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
        else:
            samplers = [None, None, None]

        train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                              batch_size=[config['batch_size_train']] + [
                                                                  config['batch_size_test']] * 2,
                                                              num_workers=[4, 4, 4],
                                                              is_trains=[True, False, False],
                                                              collate_fns=[getattr(train_dataset, 'collate_fn', None), None, None])

        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        accumulate_steps = int(config.get('accumulate_steps', 1))
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size/(config['batch_size_train']*world_size) / accumulate_steps)
        arg_sche['min_rate'] = config['min_lr'] / arg_opt['lr'] if 'min_lr' in config else 0
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        checkpointer = Checkpointer(args.output_dir)

        max_epoch = config['schedular']['epochs']
        best = 0
        best_epoch = 0

        if config['image_res'] == 224:
            print("Zero-Shot Evaluating...", flush=True)
            score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

            if utils.is_main_process():
                test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
                print(test_result, flush=True)

            dist.barrier()
            exit()

        for epoch in range(0, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config)

            # score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
            if epoch % config["eval_epoch"] == 0:
                score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

                if utils.is_main_process():
                    # val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)
                    # print(val_result)
                    test_result = itm_eval(score_test_t2i, test_loader.dataset.img2person, test_loader.dataset.txt2person, eval_mAP=True)
                    print(test_result)

                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                # **{f'val_{k}': v for k, v in val_result.items()},
                                **{f'test_{k}': v for k, v in test_result.items()},
                                'epoch': epoch}

                    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    if args.pick_best_r1:
                        score = test_result['r1']
                    elif args.pick_best_t2v:
                        score = test_result['img_r_mean']
                    else:
                        # score = test_result['r_mean']
                        score = test_result['r1']


                    if score > best:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            # 'optimizer': optimizer.state_dict(),
                            # 'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            # 'epoch': epoch,
                        }
                        checkpointer.save_checkpoint(model_state=save_obj,
                                                    epoch='best', training_states=optimizer.state_dict())

                        # torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                        best = score
                        best_epoch = epoch

                    elif epoch >= config['schedular']['epochs'] - 1:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            # 'optimizer': optimizer.state_dict(),
                            # 'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            # 'epoch': epoch,
                        }
                        checkpointer.save_checkpoint(model_state=save_obj,
                                                    epoch=epoch, training_states=optimizer.state_dict())

                        # torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))

            dist.barrier()
            torch.cuda.empty_cache()

        if utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write("best epoch: %d" % best_epoch)

            os.system(f"cat {args.output_dir}/log.txt")
            if len(args.output_hdfs) > 0:
                os.system(f'hdfs dfs -put {args.output_dir}/* {args.output_hdfs}/')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)  # this script works for both mscoco and flickr30k
    parser.add_argument('--output_hdfs', type=str, default='', help="copy to hdfs")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--local-rank', type=int, required=False)

    parser.add_argument('--epoch', default=-1, type=int)
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--k_test', default=-1, type=int, help="for evaluation")
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--override_cfg', default="", type=str, help="Use ; to separate keys")

    parser.add_argument('--pick_best_r1', action='store_true', help="save best ckpt by r@1")
    parser.add_argument('--pick_best_t2v', action='store_true', help="save best ckpt by img recall")

    args = parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    utils.update_config(config, args.override_cfg)
    if utils.is_main_process():
        print('config:', json.dumps(config))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    if len(args.output_hdfs):
        hmkdir(args.output_hdfs)
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    main(args, config)
