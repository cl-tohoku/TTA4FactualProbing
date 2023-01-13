from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def plot_dendrogram(
        dist,
        labels,
        outfile=None,
        method="centroid",
        figsize=(10, 10),
        font_size=10,
        cmap='magma_r',
        ):
    from scipy.cluster import hierarchy
    fig = plt.figure(figsize=figsize)
    # dendrogram
    axdendro = fig.add_axes([0.05, 0.05, 0.16, 0.9])
    axdendro.set_xticks([])
    axdendro.set_yticks([])
    Y = hierarchy.linkage(dist, method=method)
    Z = hierarchy.dendrogram(
        Y, orientation='left', labels=labels, leaf_font_size=font_size)
    axdendro.set_axis_off()
    # distance matrix
    index = Z['leaves']
    new_labels = [labels[i] for i in index]
    D = dist[index, :]
    D = D[:, index]
    print(D)
    axmatrix = fig.add_axes([0.3, 0.05, 0.6, 0.9])
    im = axmatrix.matshow(D.tolist(), aspect='auto', origin='lower', cmap=cmap)
    xaxis = np.arange(len(new_labels))
    axmatrix.set_xticks(xaxis)
    axmatrix.set_yticks(xaxis)
    axmatrix.set_xticklabels(new_labels)
    axmatrix.set_yticklabels(new_labels)
    axmatrix.invert_xaxis()
    # colorbar
    axcolor = fig.add_axes([0.91, 0.05, 0.02, 0.9])
    plt.colorbar(im, cax=axcolor)

    if outfile:
        fig.savefig(str(outfile))
    else:
        fig.show()
    plt.close(fig)

def create_correl_matri_img():
    args_list = [
        {
            "source_filepath": "cache/v2.11d.5/augmentations.csv",
            "img_path": "analysis/img/correl/t5-small.png"
        },
        {
            "source_filepath": "cache/v2.11d.2/augmentations.csv",
            "img_path": "analysis/img/correl/t5-large.png"
        },
        {
            "source_filepath": "cache/v2.11d.3/augmentations.csv",
            "img_path": "analysis/img/correl/t5-3b.png"
        },
        {
            "source_filepath": "cache/v2.11d.4/augmentations.csv",
            "img_path": "analysis/img/correl/t5-11b.png"
        },
        {
            "source_filepath": "cache/v2.11d.flan/augmentations.csv",
            "img_path": "analysis/img/correl/flan-xl.png"
        },
        {
            "source_filepath": "cache/v2.11d.flan.uncased/augmentations.csv",
            "img_path": "analysis/img/correl/flan-xl.uncased.png"
        },
        {
            "source_filepath": "cache/v2.11d.t03b/augmentations.csv",
            "img_path": "analysis/img/correl/t03b.png"
        },
        {
            "source_filepath": "cache/v2.11d.flan-small.uncased/augmentations.csv",
            "img_path": "analysis/img/correl/flan-small.uncased.png"
        },
    ]

    for args in args_list:
        df = pd.read_csv(args["source_filepath"])
        n_facts = len(df.index)
        df = df.T
        types = ["c_original", "c_wordswap", "c_wordnet", "c_bt-fr", "c_bt-ru", "c_bt-de", "c_bt-es", "c_bt-ja", "c_hero"]
        df = df.loc[df.index.isin(types)]
        df = df * 1.0
        print(df)

        arr = df.to_numpy()
        labels = df.index
        labels = [s[2:] for s in labels]

        correl = np.corrcoef(arr.astype(float))
        distance = 1.0 - (correl + 1.0)/2.0
        # print(distance)
        plot_dendrogram(distance, labels, outfile=args["img_path"])

if __name__ == "__main__":
    create_correl_matri_img()