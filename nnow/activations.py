import torch
from torch.autograd import Function
import torch.nn as nn


def threshold_and_support(z):
    sorted_z, _ = torch.sort(z, descending=True, dim=1)
    z_sum = sorted_z.cumsum(1) - 1  # sort of a misnomer
    support = torch.arange(1, sorted_z.size(1) + 1) * sorted_z > z_sum
    k_z = support.sum(dim=1)
    tau_z = z_sum.gather(1, k_z.unsqueeze(1) - 1).squeeze() / k_z.float()
    return tau_z.unsqueeze(1), k_z


class SparsemaxFunction(Function):

    @staticmethod
    def forward(ctx, z):
        """
        z (FloatTensor) batch x hidden size
        (long term: it would be great if it handled any shape)
        """
        tau_z, k_z = threshold_and_support(z)
        output = torch.clamp(z - tau_z, min=0)
        ctx.save_for_backward(k_z, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        k_z, output = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = (grad_input.sum(dim=1) / k_z).unsqueeze(1)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input

sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):

    def __init__(self):
        super(Sparsemax, self).__init__()

    def forward(self, input):
        # TODO: sparsemax along other dimensions
        return sparsemax(input)
