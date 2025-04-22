import torch

def human_format(num: int):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def get_model_size(model: torch.nn.Module):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f'Model params {human_format(trainable_params)}'