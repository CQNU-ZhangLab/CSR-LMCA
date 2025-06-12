import torch
from torch import optim
from torch.optim.lr_scheduler import LRScheduler
import math
from transformers import get_scheduler


def build_optimizer(args, model):

    ed_params = list(map(id, model.encoder_decoder.parameters()))
    ve_params = [p for p in model.parameters() if id(p) not in ed_params]

    param_groups = [
        {'params': ve_params, 'lr': args.lr_ve},
        {'params': [p for p in model.encoder_decoder.parameters()], 'lr': args.lr_ed}  # 直接从 model.encoder_decoder 获取
    ]

    optimizer = getattr(torch.optim, args.optim)(
        param_groups,
        betas=args.adam_betas,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )

    for i, param_group in enumerate(optimizer.param_groups, 1):
        params_count = sum(p.numel() for p in param_group['params'] if p.requires_grad)
        lr = param_group['lr']
        print(f"Parameter Group {i}: Learning Rate={lr}, Parameters Count={params_count}")

    return optimizer

def build_lr_scheduler(args, optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    return lr_scheduler

  # add cos_lr_scheduler
def build_cos_lr_scheduler(optimizer, steps, num_warmup_steps=2000, min_lr=1e-6):
    # return get_scheduler(name='cosine', optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=steps)
    lr_scheduler = CustomCosineScheduler(optimizer, num_warmup_steps=num_warmup_steps, num_cosine_steps=steps, min_lr=min_lr)
    return lr_scheduler

class CustomCosineScheduler(LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, num_cosine_steps, min_lr: float = 0, last_epoch=-1, verbose=False):
        self.num_warmup_steps = num_warmup_steps
        self.num_cosine_steps = num_cosine_steps
        self.min_lr = min_lr
        super(CustomCosineScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            # Linear warm-up phase
            return [base_lr * (self.last_epoch / self.num_warmup_steps) for base_lr in self.base_lrs]
        elif self.last_epoch <= self.num_warmup_steps + self.num_cosine_steps:
            # Cosine annealing phase
            current_step = self.last_epoch - self.num_warmup_steps
            max_steps = self.num_cosine_steps
            return [self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * current_step / max_steps))
                    for base_lr in self.base_lrs]
        else:
            # Hold at minimum learning rate
            return [self.min_lr for _ in self.base_lrs]

          
def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']


class NoamOpt(object):
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def state_dict(self):
        state_dict = self.optimizer.state_dict()
        state_dict['_step'] = self._step
        return state_dict

    def load_state_dict(self, state_dict):
        if '_step' in state_dict:
            self._step = state_dict['_step']
            del state_dict['_step']
        self.optimizer.load_state_dict(state_dict)


def get_std_opt(model, optim_func='adam', factor=1, warmup=2000):
    optim_func = dict(Adam=torch.optim.Adam,
                      AdamW=torch.optim.AdamW)[optim_func]
    return NoamOpt(model.d_model, factor, warmup,
                   optim_func(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def build_noamopt_optimizer(args, model):
    ve_optimizer = getattr(torch.optim, args.optim)(
        model.visual_extractor.parameters(),
        lr=0,
        betas=args.adam_betas,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    ed_optimizer = get_std_opt(model.encoder_decoder, optim_func=args.optim, factor=args.noamopt_factor,
                               warmup=args.noamopt_warmup)
    return ve_optimizer, ed_optimizer


class ReduceLROnPlateau(object):
    "Optim wrapper that implements rate."

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode=mode, factor=factor,
                                                              patience=patience, verbose=verbose, threshold=threshold,
                                                              threshold_mode=threshold_mode, cooldown=cooldown,
                                                              min_lr=min_lr, eps=eps)
        self.optimizer = optimizer
        self.current_lr = get_lr(optimizer)

    def step(self):
        "Update parameters and rate"
        self.optimizer.step()

    def scheduler_step(self, val):
        self.scheduler.step(val)
        self.current_lr = get_lr(self.optimizer)

    def state_dict(self):
        return {'current_lr': self.current_lr,
                'scheduler_state_dict': self.scheduler.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        if 'current_lr' not in state_dict:
            # it's normal optimizer
            self.optimizer.load_state_dict(state_dict)
            set_lr(self.optimizer, self.current_lr)  # use the lr fromt the option
        else:
            # it's a schduler
            self.current_lr = state_dict['current_lr']
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            # current_lr is actually useless in this case

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)


def build_plateau_optimizer(args, model):
    ve_optimizer = getattr(torch.optim, args.optim)(
        model.visual_extractor.parameters(),
        lr=args.lr_ve,
        betas=args.adam_betas,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    ve_optimizer = ReduceLROnPlateau(ve_optimizer,
                                     factor=args.reduce_on_plateau_factor,
                                     patience=args.reduce_on_plateau_patience)
    ed_optimizer = getattr(torch.optim, args.optim)(
        model.encoder_decoder.parameters(),
        lr=args.lr_ed,
        betas=args.adam_betas,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    ed_optimizer = ReduceLROnPlateau(ed_optimizer,
                                     factor=args.reduce_on_plateau_factor,
                                     patience=args.reduce_on_plateau_patience)

    return ve_optimizer, ed_optimizer
