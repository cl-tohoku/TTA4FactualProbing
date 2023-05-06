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
    "chatgpt":10,
    "gpt3":10
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
        if num_prompts == 30:
            num_prompts = -1
            
    return num_prompts

def main():

    file_prefix_list = [
        "t5-small",
        "t5-large",
        "t5-3b",
        "t5-11b",
        "flan-small",
        "flan-xl",
        "t03b",
        # "t5-small.chatgpt",
        # "t5-large.chatgpt",
        # "t5-small.gpt3",
        # "t5-large.gpt3",
        # "flan-small.gpt3",
        # "flan-large.gpt3"
    ]

    fig = plt.figure(figsize = (10, 5))
    ax1 = fig.add_axes([0.15, 0.20, 0.58, 0.75])
    for file_prefix in file_prefix_list:
        src = "analysis/table/sorth_eff/"+file_prefix+".csv"
        df = pd.read_table(src, index_col=0)
        df["n_prompts"] = df.apply(lambda row: get_num_prompts(row.name), axis = 1)
        df = df.loc[df['n_prompts'] > 0]
        df = df.groupby("n_prompts").mean()
        x = df.index
        y = df["total"]
        plt.plot(x, y, label = file_prefix)
        # z = np.polyfit(x, y, 1)
        # p = np.poly1d(z)
        # plt.plot(x,p(x),"--")
    
    ax1.set_xlabel("Number of prompts",size=24)
    ax1.set_ylabel("Relative Effect",size=24)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, framealpha=0.3, fontsize=18)
    dest = "analysis/img/n_prompts/ave.png"
    plt.savefig(dest, transparent=True)


if __name__ == "__main__":
    main()