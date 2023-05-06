import pandas as pd

def main():
    args_list = [
        ## ALL
        {
            "source_filepath": "cache/v2.11d.5/augmentation_summary.csv",
            "dest_filepath": "analysis/table/sorth/t5-small.csv"
        },
        {
            "source_filepath": "cache/v2.11d.2/augmentation_summary.csv",
            "dest_filepath": "analysis/table/sorth/t5-large.csv"
        },
        {
            "source_filepath": "cache/v2.11d.3/augmentation_summary.csv",
            "dest_filepath": "analysis/table/sorth/t5-3b.csv"
        },
        {
            "source_filepath": "cache/v2.11d.4/augmentation_summary.csv",
            "dest_filepath": "analysis/table/sorth/t5-11b.csv"
        },
        {
            "source_filepath": "cache/v2.11d.flan.uncased/augmentation_summary.csv",
            "dest_filepath": "analysis/table/sorth/flan-xl.csv"
        },
        {
            "source_filepath": "cache/v2.11d.t03b/augmentation_summary.csv",
            "dest_filepath": "analysis/table/sorth/t03b.csv"
        },
        {
            "source_filepath": "cache/v2.11d.flan-small.uncased/augmentation_summary.csv",
            "dest_filepath": "analysis/table/sorth/flan-small.csv"
        },
        # {
        #     "source_filepath": "cache/v2.11d.chatgpt/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/t5-small.chatgpt.csv"
        # },
        # {
        #     "source_filepath": "cache/v2.11d.chatgpt.large/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/t5-large.chatgpt.csv"
        # },
        # {
        #     "source_filepath": "cache/v2.11d.gpt3.1/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/t5-small.gpt3.csv"
        # },
        # {
        #     "source_filepath": "cache/v2.11d.gpt3.2/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/t5-large.gpt3.csv"
        # },
        # {
        #     "source_filepath": "cache/v2.11d.gpt3.3/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/flan-small.gpt3.csv"
        # },
        # {
        #     "source_filepath": "cache/v2.11d.gpt3.4/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/flan-large.gpt3.csv"
        # },

        ## DEV
        # {
        #     "source_filepath": "cache/t5-small.dev/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/t5-small.dev.csv"
        # },
        # {
        #     "source_filepath": "cache/t5-large.dev/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/t5-large.dev.csv"
        # },
        # {
        #     "source_filepath": "cache/t5-3b.dev/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/t5-3b.dev.csv"
        # },
        # {
        #     "source_filepath": "cache/t5-11b.dev/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/t5-11b.dev.csv"
        # },
        # {
        #     "source_filepath": "cache/flan-small.dev/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/flan-small.dev.csv"
        # },
        # {
        #     "source_filepath": "cache/flan-xl.dev/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/flan-xl.dev.csv"
        # },
        # {
        #     "source_filepath": "cache/t03b.dev/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/t03b.dev.csv"
        # },

        ## TEST
        # {
        #     "source_filepath": "cache/t5-small.test/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/t5-small.test.csv"
        # },
        # {
        #     "source_filepath": "cache/t5-large.test/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/t5-large.test.csv"
        # },
        # {
        #     "source_filepath": "cache/t5-3b.test/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/t5-3b.test.csv"
        # },
        # {
        #     "source_filepath": "cache/t5-11b.test/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/t5-11b.test.csv"
        # },
        # {
        #     "source_filepath": "cache/flan-small.test/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/flan-small.test.csv"
        # },
        # {
        #     "source_filepath": "cache/flan-xl.test/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/flan-xl.test.csv"
        # },
        # {
        #     "source_filepath": "cache/t03b.test/augmentation_summary.csv",
        #     "dest_filepath": "analysis/table/sorth/t03b.test.csv"
        # },
    ]

    for args in args_list:
        df = pd.read_table(args["source_filepath"], index_col=0).T
        df = df.sort_values(["c_original"]).T
        print(df)
        # df = df.loc[:, ~df.columns.isin(['P155', 'P156'])]
        df.to_csv(args["dest_filepath"], sep="\t")

if __name__ == "__main__":
    main()