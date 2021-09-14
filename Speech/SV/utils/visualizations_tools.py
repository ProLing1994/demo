import matplotlib
matplotlib.use('agg')    
import matplotlib.pyplot as plt
import numpy as np
import umap

colormap = np.array([
    [76, 255, 0],
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
], dtype=np.float) / 255 


def draw_projections(embeds, epoch, out_fpath=None, max_speakers=10):
    # init 
    speakers_per_batch = embeds.shape[0]
    utterances_per_speaker = embeds.shape[1]

    max_speakers = min(max_speakers, len(colormap))
    embeds = embeds.reshape((speakers_per_batch * utterances_per_speaker, -1))
    embeds = embeds[:max_speakers * utterances_per_speaker]
    
    n_speakers = len(embeds) // utterances_per_speaker
    ground_truth = np.repeat(np.arange(n_speakers), utterances_per_speaker)
    colors = [colormap[i] for i in ground_truth]
    
    reducer = umap.UMAP()
    projected = reducer.fit_transform(embeds)
    plt.scatter(projected[:, 0], projected[:, 1], c=colors)
    plt.gca().set_aspect("equal", "datalim")
    plt.title("UMAP projection (epoch %d)" % epoch)
    if out_fpath is not None:
        plt.savefig(out_fpath)
    plt.clf()