import pandas as pd
import matplotlib.pyplot as plt

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

    label_dict = {
        "c_hero": "no-stopwords",
        "c_bt-ja": "bt-ja",
        "c_bt-fr": "bt-fr",
        "c_bt-ru": "bt-ru",
        "c_bt-de": "bt-de",
        "c_bt-es": "bt-es",
        "c_wordnet": "wordnet",
        "c_wordswap": "embedding",

    }
    fig = plt.figure(figsize = (5, 2.5))
    ax1 = fig.add_axes([0.15, 0.34, 0.58, 0.61])
    for file_prefix in file_prefix_list:
        src = "analysis/table/sorth/"+file_prefix+".csv"
        df = pd.read_table(src, index_col=0)
        df["total"] = df.apply(lambda row: sum(row), axis = 1)

        df = df.loc[~df.index.str.contains("c_-") & ~ df.index.str.contains("all") & ~df.index.str.contains("weight")]
        df["total"] = df["total"] / df.loc["c_original", "total"]
        df = df.loc[~df.index.str.contains("original")]
        df = df.rename(label_dict)
        x = df.index
        y = df["total"]
        ax1.scatter(x, y, s=1, label = file_prefix)
        print(df)

    ax1.set_ylabel("Effectiveness",size=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    ax1.set_xticklabels(df.index.tolist())
    fig.autofmt_xdate(rotation=45)
    dest = "analysis/img/nine-types.png"
    plt.savefig(dest)

if __name__ == "__main__":
    main()