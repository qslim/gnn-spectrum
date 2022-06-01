from torch.optim.lr_scheduler import _LRScheduler
import warnings
from collections import Counter


def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch**3 * init_lr / num_batch_warm_up**3


class MultiStepLRWarmUp(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma, num_warm_up, init_lr, last_epoch=-1, verbose=False):
        self.num_warm_up = num_warm_up
        self.init_lr = init_lr
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super(MultiStepLRWarmUp, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        assert self.last_epoch + 1 == self._step_count

        if self.last_epoch + 1 <= self.num_warm_up:
            return [(self.last_epoch + 1) ** 3 * self.init_lr / self.num_warm_up ** 3 for group in self.optimizer.param_groups]
        elif self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False