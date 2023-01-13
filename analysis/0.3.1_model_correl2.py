import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_dendrogram(
        dist,
        labels,
        outfile=None,
        method="centroid",
        figsize=(6, 3),
        font_size=10,
        cmap='magma_r',
        ):
    from scipy.cluster import hierarchy
    fig = plt.figure(figsize=figsize)

    axmatrix = fig.add_axes([0.15, 0.05, 0.75, 0.85])
    im = axmatrix.matshow(dist.tolist(), aspect='auto', origin='lower', cmap=cmap)
    xaxis = np.arange(len(labels))
    axmatrix.set_xticks(xaxis)
    axmatrix.set_yticks(xaxis)
    axmatrix.set_xticklabels(labels)
    axmatrix.set_yticklabels(labels)
    axmatrix.invert_yaxis()
    # colorbar
    axcolor = fig.add_axes([0.91, 0.05, 0.02, 0.85])
    plt.colorbar(im, cax=axcolor)

    if outfile:
        fig.savefig(str(outfile))
    else:
        fig.show()
    plt.close(fig)

def main():
    file_prefix_list = [
        "t5-small",
        "t5-large",
        "t5-3b",
        "t5-11b",
        "flan-small",
        "flan-xl",
        "t03b",
    ]

    parent_df = pd.DataFrame()

    for file_prefix in file_prefix_list:
        src = "analysis/table/sorth_eff/"+file_prefix+".csv"
        df = pd.read_table(src, index_col=0)
        df = df.loc[~df.index.str.contains("weight")]
        parent_df[file_prefix] = df["total"]

    df = parent_df.T
    arr = df.to_numpy()
    labels = df.index

    correl = np.corrcoef(arr.astype(float))
    distance = 1.0 - (correl + 1.0)/2.0

    print(correl)

    plot_dendrogram(distance, labels, outfile="analysis/img/7models_eff.png")


if __name__ == "__main__":
    main()