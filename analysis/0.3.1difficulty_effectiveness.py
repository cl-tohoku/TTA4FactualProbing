import matplotlib.pyplot as plt
import pandas as pd

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
    img_dst = "analysis/img/difficulty_effectiveness.png"
    fig = plt.figure(figsize = (5, 2.5))
    ax1 = fig.add_axes([0.15, 0.2, 0.75, 0.75])

    for file_prefix in file_prefix_list:
        src_x = "analysis/table/sorth/"+file_prefix+".csv"
        src_y = "analysis/table/sorth_eff/"+file_prefix+".csv"

        ## x axis
        df = pd.read_table(src_x, index_col=0).loc["c_original"]
        df_x = (df + 1) / 501
        # df_x["total"] = (df.sum() + 1 )/(500 * len(df.index) + 1)
        print(df_x)


        ## y axis = effectiveness
        df_y = pd.read_table(src_y, index_col=0)
        df_y = df_y.loc[~df_y.index.str.contains("weight") & ~df_y.index.str.contains("c_original")].T
        df_y = df_y.loc[~df_y.index.str.contains("total")]
        print(df_y)

        df = pd.concat([df_x, df_y], axis=1)
        df = df.set_index('c_original')
        columns = df.columns.to_list()
        df = df.reset_index()
        print(df)
        print('c_original' in columns)
        
        for col in columns:
            df.plot(x='c_original', y=col, kind='scatter',s = 1, ax=ax1)
    plt.hlines([1], 0,1, "blue", linestyles='dashed')
    ax1.set_xlabel("Acc without TTA",size=12)
    ax1.set_ylabel("Effectiveness",size=12)
    plt.savefig(img_dst)


if __name__ == "__main__":
    main()