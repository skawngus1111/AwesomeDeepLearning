import torch

def calculate_top1_error(output, target) :
    _, rank1 = torch.max(output, 1)
    correct_top1 = (rank1 == target).sum().item()

    return correct_top1

def calculate_top5_error(output, target) :
    _, top5 = output.topk(5, 1, True, True)
    top5 = top5.t()
    correct5 = top5.eq(target.view(1, -1).expand_as(top5))

    for k in range(6):
        correct_k = correct5[:k].reshape(-1).float().sum(0, keepdim=True)

    correct_top5 = correct_k.item()

    return correct_top5