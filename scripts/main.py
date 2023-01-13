import pandas as pd

from data import ExpResult, Facts, Prompts, Generations, CorrectTable
from settings import get_setting

# name = "v2.6d.6"
# name = "test-flan"
name = "v2.11d.5"
# name = "lama-lb"

# Augment + Infer
settings = get_setting(name)
exp_result = ExpResult(settings)

# Aggregate
exp_result.aggregate(settings["aggregation_types"])

exp_result.facts.get_corrects_table()
correct_table = CorrectTable(facts_df=exp_result.facts.df, prompts=exp_result.prompts)
correct_table.df.to_csv("cache/{}/iscorrect.csv".format(name))
# aug_effect = correct_table.compare_original_to_augment()


exp_result.facts.df.to_csv("cache/{}/facts.csv".format(name))
exp_result.prompts.df.to_csv("cache/{}/prompts.csv".format(name))
# exp_result.prompts.df.to_csv("cache/{}/prompts.csv".format(name))
# exp_result.generations.df.to_csv("cache/{}/generations.csv".format(name))

