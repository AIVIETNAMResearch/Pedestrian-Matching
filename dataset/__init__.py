import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.transforms import InterpolationMode
from dataset.retrieval_dataset import ps_eval_dataset, ps_train_dataset

from dataset.randaugment import RandomAugment
from dataset.random_erasing import RandomErasing

import timm

def create_dataset(dataset, config, evaluate=False):

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    train_transform_ps_weak = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    # train_transform_ps_cnn = transforms.Compose([
    #     transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    data_config = {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 
                   'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 
                   'crop_pct': 0.875, 'crop_mode': 'center'}
    train_transform_ps_cnn = timm.data.create_transform(**data_config, is_training=False)
    

    train_transform_ps_strong = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),

        transforms.ToTensor(),
        normalize,
        RandomErasing(probability=config['erasing_p'], mean=[0.0, 0.0, 0.0])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'ps':
        test_dataset = ps_eval_dataset(config['test_file'], test_transform, config['image_root'], config["max_tokens"])
        if evaluate:
            return None, None, test_dataset

        train_dataset = ps_train_dataset(config['train_file'], train_transform_ps_weak, train_transform_ps_strong, config['image_root'], config["max_tokens"], transform_cnn=train_transform_ps_cnn)
        val_dataset = ps_eval_dataset(config['val_file'], test_transform, config['image_root'], config["max_tokens"])

        return train_dataset, val_dataset, test_dataset
    else:
        raise NotImplementedError(f"dataset == {dataset}")


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def vqa_classify_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights, _ in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), question_list, torch.Tensor(answer_list), torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

    return loaders

def vqa_mc_collate_fn(batch):
    image_list, question_list, answer_list = [], [], []
    cand_list = [ [], [], [], [], [] ]
    for image, question, answer, cand_ in batch:
        image_list.append(image)
        question_list.append(question)     
        answer_list.append(int(answer))
        # cand_list += cand_
        # print(cand_)
        for i in range(5):
            cand_list[i].append(cand_[i])
    return torch.stack(image_list, dim=0), question_list, torch.Tensor(answer_list), cand_list