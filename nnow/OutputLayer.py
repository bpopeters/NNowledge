import torch.nn as nn
import torch.nn.functional as F

class LogSoftmaxOutput(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(LogSoftmaxOutput, self).__init__()
        self.affine = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_layer):
        return F.log_softmax(self.affine(hidden_layer), dim=1)
