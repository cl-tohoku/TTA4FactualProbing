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

    fig = plt.figure(figsize = (5, 2.5))
    ax1 = fig.add_axes([0.15, 0.20, 0.58, 0.75])
    for file_prefix in file_prefix_list:
        src = "analysis/table/sorth_eff/"+file_prefix+".csv"
        df = pd.read_table(src, index_col=0)
        df["n_prompts"] = df.apply(lambda row: get_num_prompts(row.name), axis = 1)
        


        x = df["n_prompts"]
        y = df["total"]
        plt.scatter(x, y, s=1, label = file_prefix)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x,p(x),"--")
    
    ax1.set_xlabel("Number of prompts",size=12)
    ax1.set_ylabel("Effectiveness",size=12)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    dest = "analysis/img/n_prompts/all.png"
    plt.savefig(dest)


if __name__ == "__main__":
    main()