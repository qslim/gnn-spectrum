import torch


class JumpingKnowledge(torch.nn.Module):

    def __init__(self, mode, weight=None):
        super(JumpingKnowledge, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['c', 's', 'm', 'l', 'w']

        self.weight = weight

    def forward(self, xs):
        assert isinstance(xs, list) or isinstance(xs, tuple)

        if self.mode == 'c':
            return torch.cat(xs, dim=-1)
        elif self.mode == 'l':
            return xs[-1]
        elif self.mode == 'w':
            nr = 0
            total_weight = 0
            for i, x in enumerate(xs):
                nr = nr + x * self.weight[i]
                total_weight = total_weight + self.weight[i]
            return nr / total_weight
        else:
            nr = 0
            for x in xs:
                nr = nr + x
            if self.mode == 'm':
                return nr / len(xs)
            else:
                return nr

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.mode)
