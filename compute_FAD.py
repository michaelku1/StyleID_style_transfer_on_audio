import os
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from scipy.linalg import sqrtm

# --------------------------
# Load audio
# --------------------------
def load_audio_files(directory, sample_rate=16000):
    waveforms = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            path = os.path.join(directory, filename)
            waveform, sr = torchaudio.load(path)
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                waveform = resampler(waveform)
            waveforms.append(waveform.mean(0).unsqueeze(0))  # Mono
    return waveforms

# --------------------------
# Get embeddings (CLAP or VGGish)
# --------------------------

def get_embeddings_clap(waveforms, model):
    embeddings = []
    for w in tqdm(waveforms, desc="Embedding with CLAP"):
        with torch.no_grad():
            emb = model.get_audio_embedding_from_data(x=w.to(model.device), use_tensor=True)
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)

def get_embeddings_vggish(waveforms, model, postprocessor):
    embeddings = []
    for w in tqdm(waveforms, desc="Embedding with VGGish"):
        emb = model(w)  # shape: (frames, 128)
        if postprocessor:
            emb = postprocessor(emb)
        embeddings.append(emb.detach().cpu().numpy().mean(axis=0))  # Average over frames
    return np.vstack(embeddings)

# --------------------------
# FAD computation
# --------------------------

def compute_fad(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)

def get_stats(embeddings):
    mu = np.mean(embeddings, axis=0)
    sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma

# --------------------------
# Main
# --------------------------

def compute_fad_for_model(model_name="CLAP"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_dir = "real_audio_dir"
    gen_dir = "generated_audio_dir"

    real_waveforms = load_audio_files(real_dir)
    gen_waveforms = load_audio_files(gen_dir)

    if model_name == "CLAP":
        from CLAPWrapper import CLAP  # See note below if this fails
        model = CLAP(version="2023", use_cuda=torch.cuda.is_available())
        real_emb = get_embeddings_clap(real_waveforms, model)
        gen_emb = get_embeddings_clap(gen_waveforms, model)

    elif model_name == "VGGish":
        import torchvggish
        from torchvggish import vggish, vggish_input, vggish_params

        model = vggish()
        model.eval().to(device)

        # Load postprocessor if needed
        postprocessor = None
        try:
            from torchvggish import vggish_postprocess
            postprocessor = vggish_postprocess.Postprocessor()
        except:
            pass

        def wrap_for_vggish(wavs):
            return [vggish_input.waveform_to_examples(w.squeeze().numpy(), sr=16000) for w in wavs]

        real_emb = get_embeddings_vggish(wrap_for_vggish(real_waveforms), model, postprocessor)
        gen_emb = get_embeddings_vggish(wrap_for_vggish(gen_waveforms), model, postprocessor)

    else:
        raise ValueError("Unknown model")

    mu1, sigma1 = get_stats(real_emb)
    mu2, sigma2 = get_stats(gen_emb)
    fad_score = compute_fad(mu1, sigma1, mu2, sigma2)
    print(f"FAD score using {model_name}: {fad_score:.4f}")

if __name__ == "__main__":
    compute_fad_for_model("CLAP")   # or "VGGish"