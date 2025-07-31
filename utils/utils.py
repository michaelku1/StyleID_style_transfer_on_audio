import os
import torch
import torchaudio
from torch import Tensor as T
import matplotlib.pyplt as plt

from typing import Dict, Optional, List, Any, Tuple, Type, Union



def plot_loss(log_dir, loss_history: List[float]):
    fig, ax = plt.subplots()
    ax.plot(loss_history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    plt.grid(c="lightgray")
    outfilepath = os.path.join(log_dir, "loss.png")
    plt.savefig(outfilepath, dpi=300)
    plt.close("all")


def plot_response(
    y: torch.Tensor,
    x_hat: torch.Tensor,
    x: torch.Tensor,
    sample_rate: int = 44100,
    epoch: int = 0,
):
    fig, ax = plt.subplots(figsize=(6, 4))

    # compute frequency response of y
    Y = torch.fft.rfft(y)
    Y = torch.abs(Y)
    Y_db = 20 * torch.log10(Y + 1e-8)

    # compute frequency response of x_hat
    X_hat = torch.fft.rfft(x_hat)
    X_hat = torch.abs(X_hat)
    X_hat_db = 20 * torch.log10(X_hat + 1e-8)

    # compute frequency response of x
    X = torch.fft.rfft(x)
    X = torch.abs(X)
    X_db = 20 * torch.log10(X + 1e-8)

    # compute frequency axis
    freqs = torch.fft.fftfreq(x.shape[-1], d=1 / sample_rate)
    freqs = freqs[: X.shape[-1] - 1]  # take only positive frequencies
    X_db = X_db[:, : X.shape[-1] - 1]
    X_hat_db = X_hat_db[:, : X_hat.shape[-1] - 1]
    Y_db = Y_db[:, : Y.shape[-1] - 1]

    # smooth frequency response
    kernel_size = 1023
    X_db = torch.nn.functional.avg_pool1d(
        X_db.unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    X_hat_db = torch.nn.functional.avg_pool1d(
        X_hat_db.unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    Y_db = torch.nn.functional.avg_pool1d(
        Y_db.unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    # plot frequency response
    ax.plot(freqs, Y_db[0].squeeze().cpu().numpy(), label="input", alpha=0.7)
    ax.plot(freqs, X_hat_db[0].cpu().squeeze().numpy(), label="pred", alpha=0.7)
    ax.plot(
        freqs,
        X_db[0].squeeze().cpu().numpy(),
        label="target",
        alpha=0.7,
        c="gray",
        linestyle="--",
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(100, 20000)
    ax.set_xscale("log")
    plt.legend()
    plt.grid(c="lightgray")
    plt.tight_layout()
    plt.savefig(f"outputs/auto_eq/audio/epoch={epoch:03d}_response.png", dpi=300)


def randint(self, low: int, high: int, n: int = 1) -> Union[int, T]:
    x = torch.randint(low=low, high=high, size=(n,))
    if n == 1:
        return x.item()
    return x

def random_index(n_retries, file_n_samples, n_samples, end_buffer_n_samples):
    '''
    randomly generate starting index for random chunking
    '''
    
    torch.manual_seed(12345)

    # assuming input audio data is >> user defined audio chunk (e.g ~2s)
    high = int(file_n_samples - n_samples - end_buffer_n_samples + 1)
    
    random_indexes = []
    for _ in range(n_retries):
        start_idx = randint(0, high)
        random_indexes.append(start_idx)

    return random_indexes

def random_index_list(n_retries, data_path, n_samples, end_buffer_n_samples):
    # a list of random indexes per file
    list_of_permuted_indexes = []
    file_list = sorted(os.listdir(data_path))
    
    for i in range(len(file_list)):
        file_list[i] = os.path.join(base_path, file_list[i])

        for input_path in file_list:
            if os.path.basename(input_path) == '.DS_Store':
                continue
            file_info = torchaudio.info(input_path)
            
            total_number_of_frames = file_info.num_frames
            # list
            random_indexes = random_index(n_retries, total_number_of_frames, n_samples, end_buffer_n_samples)
            list_of_permuted_indexes.append(random_indexes)