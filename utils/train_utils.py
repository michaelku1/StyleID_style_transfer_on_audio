import torch

def compute_mean_loss(loss_history):
    """
    compute mean loss for each loss (e.g L1, MSTFT)

    input:
        loss_history: list of dictionaries, each dictionary contains the loss values for each loss function
    output:
        mean_loss_history: dictionary, each key is the loss function name, and the value is the mean loss value
    """

    mean_loss_history = {} # {loss_name_1: [loss_value1, loss_value2, ...], loss_name_2: [loss_value1, loss_value2, ...], ...}
    for loss_dict_item in loss_history:
        for loss_name, loss_value in loss_dict_item.items():
            if loss_name not in mean_loss_history:
                mean_loss_history[loss_name] = []
            mean_loss_history[loss_name].append(loss_value)
    
    for loss_name, list_of_loss_values in mean_loss_history.items():
        mean_loss_history[loss_name] = torch.mean(torch.stack(list_of_loss_values), dim=0)

    return mean_loss_history

loss_history = [
    {"L1": torch.tensor([1.0, 2.0]), "MSTFT": torch.tensor([2.0, 3.0])},
    {"L1": torch.tensor([3.0, 4.0]), "MSTFT": torch.tensor([4.0, 5.0])},
]

print(compute_mean_loss(loss_history))