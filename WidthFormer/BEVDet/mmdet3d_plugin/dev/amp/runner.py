# Auto Mixed Precision Trainer
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from mmcv.runner.epoch_based_runner import EpochBasedRunner
from mmcv.runner.iter_based_runner import IterBasedRunner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.utils import get_host_info
from mmcv.runner import OptimizerHook
from mmcv.runner.hooks.optimizer import Fp16OptimizerHook, GradientCumulativeFp16OptimizerHook

# from mmcls.core.utils.dist_utils import DistOptimizerHook


class DistOptimizerHook(OptimizerHook):

    def __init__(self, grad_clip=None, coalesce=True, bucket_size_mb=-1):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
        runner.optimizer.step()

@RUNNERS.register_module()
class AmpEpochBasedRunner(EpochBasedRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

    def _amp_train_step(self, data_batch, **kwargs):
        with autocast():
            self.run_iter(data_batch, train_mode=True, **kwargs)
            losses = self.outputs['loss']
        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()
        if self.grad_clip is not None:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip['max_norm'])

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self._amp_train_step(data_batch, **kwargs)
            self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def run(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,
            **kwargs) -> None:

        self.grad_clip = None
        _hooks = []
        for hook in self._hooks:
            if isinstance(hook, Fp16OptimizerHook) or \
               isinstance(hook, GradientCumulativeFp16OptimizerHook):
                raise AttributeError('MMCV based FP16 is not supported by %s' % self.__class__.__name__)
            elif isinstance(hook, DistOptimizerHook):
                self.grad_clip = hook.grad_clip
            elif isinstance(hook, OptimizerHook):
                self.grad_clip = hook.grad_clip
            elif not isinstance(hook, DistOptimizerHook):
                _hooks.append(hook)
        self._hooks = _hooks

        super(AmpEpochBasedRunner, self).run(data_loaders, workflow, max_epochs, **kwargs)


@RUNNERS.register_module()
class AmpIterBasedRunner(IterBasedRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

    def _amp_train_step(self, data_batch, **kwargs):
        with autocast():
            outputs = self.model.train_step(data_batch, None, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('model.train_step() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
            self.outputs = outputs
            losses = self.outputs['loss']
        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()
        if self.grad_clip is not None:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip['max_norm'])

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.call_hook('before_train_iter')
        self._amp_train_step(data_batch, **kwargs)
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1

    def run(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,
            **kwargs) -> None:

        self.grad_clip = None

        _hooks = []
        for hook in self._hooks:
            if isinstance(hook, Fp16OptimizerHook) or \
               isinstance(hook, GradientCumulativeFp16OptimizerHook):
                raise AttributeError('MMCV based FP16 is not supported by %s' % self.__class__.__name__)
            elif isinstance(hook, DistOptimizerHook):
                self.grad_clip = hook.grad_clip
            elif isinstance(hook, OptimizerHook):
                self.grad_clip = hook.grad_clip
            elif not isinstance(hook, DistOptimizerHook):
                _hooks.append(hook)
        self._hooks = _hooks

        super(AmpIterBasedRunner, self).run(data_loaders, workflow, max_epochs, **kwargs)
