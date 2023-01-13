import matplotlib.pyplot as plt
import pandas as pd

def graph_acc():
    args_list = [
        {
            "source_filepath1": "analysis/table/sorth_acc/t5-small.csv",
            "source_filepath2": "analysis/table/sorth/t5-small.csv",
            "dst_filepath": "analysis/img/sorth_acc/t5-small.png"
        },
        {
            "source_filepath1": "analysis/table/sorth_acc/t5-large.csv",
            "source_filepath2": "analysis/table/sorth/t5-large.csv",    
            "dst_filepath": "analysis/img/sorth_acc/t5-large.png"
        },
        {
            "source_filepath1": "analysis/table/sorth_acc/t5-3b.csv",
            "source_filepath2": "analysis/table/sorth/t5-3b.csv",
            "dst_filepath": "analysis/img/sorth_acc/t5-3b.png"
        },
        {
            "source_filepath1": "analysis/table/sorth_acc/t5-11b.csv",
            "source_filepath2": "analysis/table/sorth/t5-11b.csv",
            "dst_filepath": "analysis/img/sorth_acc/t5-11b.png"
        },
        {
            "source_filepath1": "analysis/table/sorth_acc/flan-xl.csv",
            "source_filepath2": "analysis/table/sorth/flan-xl.csv",
            "dst_filepath": "analysis/img/sorth_acc/flan-xl.png"
        },
        {
            "source_filepath1": "analysis/table/sorth_acc/flan-xl.uncased.csv",
            "source_filepath2": "analysis/table/sorth/flan-xl.uncased.csv",
            "dst_filepath": "analysis/img/sorth_acc/flan-xl.uncased.png"
        },
        {
            "source_filepath1": "analysis/table/sorth_acc/t03b.csv",
            "source_filepath2": "analysis/table/sorth/t03b.csv",
            "dst_filepath": "analysis/img/sorth_acc/t03b.png"
        },
        {
            "source_filepath1": "analysis/table/sorth_acc/flan-small.uncased.csv",
            "source_filepath2": "analysis/table/sorth/flan-small.uncased.csv",
            "dst_filepath": "analysis/img/sorth_acc/flan-small.uncased.png"
        },
    ]

    for args in args_list:
        df1 = pd.read_table(args["source_filepath1"], index_col=0).T
        df2 = pd.read_table(args["source_filepath2"], index_col=0).loc["c_original"]
        fig = plt.figure(figsize = (5, 2.5))
        ax1 = fig.add_axes([0.15, 0.3, 0.68, 0.65])
        ax2 = ax1.twinx()
        df1.plot(legend=False, ax = ax1)
        df2.plot(legend=False, kind='bar', ax = ax2)
        ax1.set_xticks(range(len(df1.index)))
        ax1.set_xticklabels(df1.index.tolist(), rotation=90)
        ax1.set_zorder(1.0)
        ax2.set_zorder(0.0)
        
        ax1.set_xlabel("Relation",size=12)
        ax2.set_ylabel("Number of corrects",size=12)
        ax1.set_ylabel("Effectiveness",size=12)
        ax1.patch.set_alpha(0)
        plt.savefig(args["dst_filepath"])
        plt.clf()

graph_acc()