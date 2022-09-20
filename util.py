import torch
import random
import logging
import numpy as np
from config import Config
from dataset import ExampleDataset
from transformers import AdamW, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split

def setup_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False  #True 可加速训练


def setup_logging(log_path, name='train.log', mode='a'):
    logging.basicConfig(
        level=logging.INFO,
        filename=log_path+name,
        filemode=mode,
        format='%(asctime)s-%(message)s'
    )


def set_data_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: set_data_device(sample, device) for k, sample in data.items()}
    elif isinstance(data, tuple):
        return [set_data_device(sample, device) for sample in data]
    else:
        return data


def build_optimizers(config: Config, model: torch.nn.Module, max_steps, use_scheduler = True):

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay':config.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    num_warmup_steps = int(config.warmup_ratio*max_steps)

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=config.learning_rate, eps=config.adam_epsilon)
    # optimizer=torch.optim.AdamW(optimizer_grouped_parameters,lr=config.learning_rate)
    if use_scheduler:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_steps)
    return optimizer, scheduler


def build_dataloaders(config: Config, mode='train'):
    dataset = ExampleDataset(config, mode)
    if mode == 'train':
        size = len(dataset)
        val_size = int(size * config.val_ratio)
        train_dataset, val_dataset = random_split(
            dataset,
            lengths=[size - val_size, val_size],
            generator=torch.Generator().manual_seed(config.seed)
        )
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            sampler=train_sampler,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.val_batch_size,
            sampler=val_sampler,
            drop_last=False,
        )
        return train_dataloader, val_dataloader
    else:
        bs = config.test_batch_size if mode == ' test' else config.pretrain_batch_size
        sampler = SequentialSampler(dataset)
        loader = DataLoader(
            dataset,
            batch_size=bs,
            sampler=sampler,
            drop_last=True if mode == 'pretrain' else False,
        )
        return loader


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + \
                    self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings.weight', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
