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

    for file_prefix in file_prefix_list:
        print(file_prefix,'----------------------')
        dev = "analysis/table/sortv/"+file_prefix+".dev.csv"
        test = "analysis/table/sortv/"+file_prefix+".test.csv"
        dst = "analysis/table/agnostic/"+file_prefix+".csv"

        dev_df = pd.read_table(dev, index_col=0)
        dev_10 = dev_df[:10][["rank"]]
        dev_10 = dev_10.rename(columns={"rank": "dev_rank"})
        # print(dev_10)
        combinations = dev_10.index.to_list()

        test_df = pd.read_table(test, index_col=0)
        test_10 = test_df.loc[test_df.index.isin(combinations)][["rank"]]
        test_10 = test_10.rename(columns={"rank": "test_rank"})
        # print(test_10)

        agnostic_df = pd.concat([dev_10, test_10], axis=1)
        agnostic_df.to_csv(dst, sep="\t")
        print(agnostic_df)
        print('\noriginal', dev_df.loc['c_original']['rank'],test_df.loc['c_original']['rank'], )
        print('\n\n')
        # print(dev_df.loc['c_original']['rank'])

        # df["total"] = df.apply(lambda row: sum(row), axis = 1)
        # df["rank"] = df["total"].rank(ascending=False)
        # df = df.sort_values("total", ascending=False)
        # df.to_csv(dst, sep="\t")
        



if __name__ == "__main__":
    main()