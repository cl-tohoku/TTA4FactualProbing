import pandas as pd
import pickle
from fuzzywuzzy import process
import os
import numpy as np

def get_original_dfs(folder):
    dataset_path = 'v2.11d.csv'
    with open('{}/evaluate_all.pkl'.format(folder), 'rb') as f:
        aggregation_result = pickle.load(f)

    aggregation_result['top_conf_all'] = aggregation_result.apply(lambda row: row['score'] / row['score_total'], axis=1)
    aggregation_result['gold_conf_all'] = aggregation_result.apply(lambda row: row['gold_score'] / row['score_total'], axis=1)

    dataset_df = pd.read_csv(dataset_path, index_col=0)
    gpt3_prompts_df = pd.read_csv(folder+'prompts.csv', index_col=0)
    dataset_df = dataset_df.iloc[dataset_df.index.repeat(11)].reset_index(drop=True)
    aggregation_result = aggregation_result.iloc[aggregation_result.index.repeat(11)].reset_index(drop=True)
    gpt3_prompts_df = pd.concat([gpt3_prompts_df, dataset_df[['ss', 'p']], aggregation_result[['top_conf_all', 'gold_conf_all']]], axis=1)
    return gpt3_prompts_df

def filter_templates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[df.fact_id % 2 == 0]
    df = df.loc[(df.type != '0original') & (df['template'] != "") & (df['template'].str.contains('[MASK]'))]
    return df

def get_templates(df) -> pd.DataFrame:
    df['template'] = df.apply(lambda row: row['prompt'].replace(row['ss'], '[MASK]') if type(row['prompt'])==str  else "", axis=1)
    return filter_templates(df)

def calc_effectivenss(df):
    # lower the better if False
    df['effect_top'] = df.apply(lambda row: row['top_conf_all'] / row['top_conf'], axis=1)
    # higher the better
    df['effect_gold'] = df.apply(lambda row: row['gold_conf_all'] / row['gold_conf'] if row['gold_conf'] != 0 else np.nan, axis=1)
    return df

def get_fuzzy_score(template, template_dict, col_label):
    choices = list(template_dict.keys())
    choices.remove(template)

    fuzzy_matched = process.extractBests(template, choices, limit=10, score_cutoff=98)
    if len(fuzzy_matched) == 0:
        return
    print(template, fuzzy_matched)
    # df[col_label] = df.apply(lambda row: fuzz.process.)
# print(dataset_df)

def join_close_templates(df: pd.DataFrame, cutoff_score: float) -> pd.DataFrame:
    print(len(df.index))
    for i in df.index:
        # if i >= 10:
        #     continue
        try:
            template = df.loc[i, 'template']
        except KeyError:
            continue

        templates = {v: k for k, v in df['template'].to_dict().items()}
        choices = list(templates.keys())
        choices.remove(template)
        join_cands = process.extractBests(template, choices, score_cutoff=cutoff_score)
        if len(join_cands) == 0:
            continue
        print(template, join_cands)

        for (cand, score) in join_cands:
            before_count = df.loc[i,'fact_id']
            df.loc[i,'fact_id'] = before_count + df.loc[templates[cand], 'fact_id']
            df.loc[i,'effect_gold'] = (before_count * df.loc[i,'effect_gold'] + df.loc[templates[cand], 'fact_id'] * df.loc[templates[cand], 'effect_gold']) / df.loc[i,'fact_id']
            df.loc[i,'effect_top'] = (before_count * df.loc[i,'effect_top']+ df.loc[templates[cand], 'fact_id'] * df.loc[templates[cand], 'effect_top']) / df.loc[i,'fact_id']
            df = df.drop(templates[cand])
    print(df.sort_values('fact_id', ascending=False).head(10))
    return df.sort_values('fact_id', ascending=False)
    # for i, row in df.iterrows():


def main(folder):
    df = get_original_dfs(folder)
    df = calc_effectivenss(df)
    df = get_templates(df)
    print(df)
    gb_template = df.groupby(['p', 'template']).aggregate({'fact_id':'count','effect_gold':'mean', 'effect_top':'mean'})
    gb_template = gb_template.sort_values(['p', 'fact_id'], ascending=[True, False])
    print(gb_template)

    df_templates = gb_template.reset_index()
    cutoff_score = 95
    print(df_templates)
    for k, d in df_templates.groupby('p'):
        # if k != 'P103':
        #     continue


        outname = f'{k}.csv'
        outdir = f'prompt_analysis/templates/{cutoff_score}'  
        os.makedirs(outdir, exist_ok=True)
        fullname = os.path.join(outdir, outname)
        if os.path.exists(fullname):
            continue

        d = join_close_templates(d, cutoff_score)

        d.to_csv(fullname)  
    

if __name__ == "__main__":
    main(folder='cache/v2.11d.gpt3.2/')