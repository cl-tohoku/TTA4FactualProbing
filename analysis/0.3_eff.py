import pandas as pd
import numpy as np

def main():
    file_prefix_list = [
        "t5-small",
        "t5-small.dev",
        "t5-small.test",
        "t5-large",
        "t5-large.dev",
        "t5-large.test",
        "t5-3b",
        "t5-3b.dev",
        "t5-3b.test",
        "t5-11b",
        "t5-11b.dev",
        "t5-11b.test",
        "flan-small",
        "flan-small.dev",
        "flan-small.test",
        "flan-xl",
        "flan-xl.dev",
        "flan-xl.test",
        "t03b",
        "t03b.dev",
        "t03b.test",
    ]

    for file_prefix in file_prefix_list:
        src = "analysis/table/sorth/"+file_prefix+".csv"
        dst = "analysis/table/sorth_eff/"+file_prefix+".csv"

        df = pd.read_table(src, index_col=0)
        df["total"] = df.apply(lambda row: sum(row), axis = 1)
        df = df.apply(lambda row: (np.asarray(row) + 1)/ (df.loc["c_original"] + 1), axis=1)
        df.to_csv(dst, sep="\t")
        



if __name__ == "__main__":
    main()

# import pandas as pd
# import numpy as np

# def get_sorth_acc():
#     args_list = [
#         {
#             "source_filepath": "analysis/table/sorth/t5-small.csv",
#             "dest_filepath": "analysis/table/sorth_acc/t5-small.csv"
#         },
#         {
#             "source_filepath": "analysis/table/sorth/t5-large.csv",
#             "dest_filepath": "analysis/table/sorth_acc/t5-large.csv"
#         },
#         {
#             "source_filepath": "analysis/table/sorth/t5-3b.csv",
#             "dest_filepath": "analysis/table/sorth_acc/t5-3b.csv"
#         },
#         {
#             "source_filepath": "analysis/table/sorth/t5-11b.csv",
#             "dest_filepath": "analysis/table/sorth_acc/t5-11b.csv"
#         },
#         {
#             "source_filepath": "analysis/table/sorth/flan-xl.csv",
#             "dest_filepath": "analysis/table/sorth_acc/flan-xl.csv"
#         },
#         {
#             "source_filepath": "analysis/table/sorth/flan-xl.uncased.csv",
#             "dest_filepath": "analysis/table/sorth_acc/flan-xl.uncased.csv"
#         },
#         {
#             "source_filepath": "analysis/table/sorth/t03b.csv",
#             "dest_filepath": "analysis/table/sorth_acc/t03b.csv"
#         },
#         {
#             "source_filepath": "analysis/table/sorth/flan-small.uncased.csv",
#             "dest_filepath": "analysis/table/sorth_acc/flan-small.uncased.csv"
#         },
#     ]

#     for args in args_list:
#         df = pd.read_table(args["source_filepath"], index_col=0)
#         df = df.apply(lambda row: (np.asarray(row) + 1)/ (df.loc["c_original"] + 1), axis=1)
#         df.to_csv(args["dest_filepath"], sep="\t")