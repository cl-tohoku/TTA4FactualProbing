import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

method_to_n = {
    "orig": 1,
    "swap": 4,
    "net": 4,
    "fr": 4,
    "ru": 4,
    "de": 4,
    "es": 4,
    "ja": 4,
    "hero": 1,
}

def get_num_prompts(col_name):
    num_prompts = 30
    if col_name == "c_all":
        pass
    elif col_name.startswith("c_-"):
        splits = col_name.split("-")
        for split in splits[1:]:
            num_prompts -= method_to_n[split]
    else:
        for key in method_to_n.keys():
            if key in col_name:
                num_prompts -= num_prompts - method_to_n[key]
            
    return num_prompts

def num_prompts_graph():

    args_list = [
        {
            "source_filepath1": "analysis/table/sorth/t5-small.csv",
            "label": "t5-small"
        },
        {
            "source_filepath1": "analysis/table/sorth/t5-large.csv", 
            "label": "t5-large"
        },
        {
            "source_filepath1": "analysis/table/sorth/t5-3b.csv",
            "label": "t5-3b"
        },
        {
            "source_filepath1": "analysis/table/sorth/t5-11b.csv",
            "label": "t5-11b"
        },
        {
            "source_filepath1": "analysis/table/sorth/flan-small.uncased.csv",
            "label": "flan-small"
        },
        # {
        #     "source_filepath1": "analysis/table/sorth/flan-xl.csv",
        #     "label": "flan-xl"
        # },
        {
            "source_filepath1": "analysis/table/sorth/flan-xl.uncased.csv",
            "label": "flan-xl"
        },
        {
            "source_filepath1": "analysis/table/sorth/t03b.csv",
            "label": "t03b"
        },
        ]
    
    for i in range(len(args_list)):
        df = pd.read_table(args_list[i]["source_filepath1"], index_col=0)
        df = df.loc[~df.index.str.contains("weight")]
        df["total"] = df.apply(lambda row: sum(row), axis = 1)
        df["rank"] = df["total"].rank(ascending=False)
        order_list = df.sort_values("rank").index
        print(order_list)

        fig = plt.figure(figsize = (5, 2.5))
        ax1 = fig.add_axes([0.15, 0.20, 0.58, 0.75])
        for args in args_list:
            print(args["label"])
            df = pd.read_table(args["source_filepath1"], index_col=0)
            df = df.loc[~df.index.str.contains("weight")]
            df["total"] = df.apply(lambda row: sum(row), axis = 1)
            df["rank"] = df["total"].rank(ascending=False)
            df = df.reindex(order_list)[:50]
            if args["label"] == "flan-xl":
                df = df.sort_values("rank")
                ax1.set_xticks(range(len(df.index)))
                ax1.set_xticklabels(df.index.tolist(), rotation=90)
            print(df)
            # df["total"] = df["total"] / df.loc["c_original", "total"]
            # print(df)
            
            # remove effect of bt-ja
            # df = df.loc[:, ["n_prompts", "total"]]
            # df = df.loc[(df.index.str.contains("-ja"))|(~df.index.str.contains("c_-"))]
            # df = df.loc[~df.index.isin(["c_bt-ja", "c_all", "c_weight"])]
            # print(df)

            # df = df.groupby("n_prompts").mean()

            x = df.index
            y = df["rank"]
            plt.scatter(x, y, s=1, label = args["label"])
            # z = np.polyfit(x, y, 1)
            # p = np.poly1d(z)
            # plt.plot(x,p(x),"--")
        # df.plot(legend=False, ax = ax1)
        # # df2.plot(legend=False, kind='bar', ax = ax2)
        # ax1.set_xticks(range(len(df1.index)))
        # ax1.set_xticklabels(df1.index.tolist(), rotation=90)
        # ax1.set_zorder(1.0)
        # ax2.set_zorder(0.0)
        ax1.set_xticks([], [])
        ax1.set_xlabel("Combinations",size=12)
        ax1.set_ylabel("Rank",size=12)
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        dest = "analysis/img/511_{}.png".format(args_list[i]["label"])
        # ax1.patch.set_alpha(0)
        plt.savefig(dest)
        # plt.clf()
        # print(df)



num_prompts_graph()