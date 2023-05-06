import pandas as pd

def main():
    file_prefix_list = [
        # "t5-small",
        # "t5-small.dev",
        # "t5-small.test",
        # "t5-large",
        # "t5-large.dev",
        # "t5-large.test",
        # "t5-3b",
        # "t5-3b.dev",
        # "t5-3b.test",
        # "t5-11b",
        # "t5-11b.dev",
        # "t5-11b.test",
        # "flan-small",
        # "flan-small.dev",
        # "flan-small.test",
        # "flan-xl",
        # "flan-xl.dev",
        # "flan-xl.test",
        # "t03b",
        # "t03b.dev",
        # "t03b.test",
        # "t5-small.chatgpt",
        # "t5-large.chatgpt",
        "t5-small.gpt3",
        "t5-large.gpt3",
        "flan-small.gpt3",
        "flan-large.gpt3"
    ]

    for file_prefix in file_prefix_list:
        src = "analysis/table/sorth/"+file_prefix+".csv"
        dst = "analysis/table/sortv/"+file_prefix+".csv"

        df = pd.read_table(src, index_col=0)
        df["total"] = df.apply(lambda row: sum(row), axis = 1)
        df["rank"] = df["total"].rank(ascending=False)
        df = df.sort_values("total", ascending=False)
        df.to_csv(dst, sep="\t")
        



if __name__ == "__main__":
    main()