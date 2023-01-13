import pandas as pd
import pickle

def get_dec_fact_index(prompts_df):
    prompts_df['is_correct'] = prompts_df['is_correct'] * 1.0
    
    original_correct_fid = prompts_df.loc[(prompts_df['type'] == '0original') & (prompts_df['is_correct'] == 1.0)]['fact_id']
    n_corrects_by_fact = prompts_df.groupby(['fact_id']).sum().sort_values('is_correct')

    dec = n_corrects_by_fact.loc[n_corrects_by_fact.index.isin(original_correct_fid)]
    return dec.index.to_list()

def main():
    prompts_path = 'cache/v2.11d.2/prompts.csv'
    prompts_df = pd.read_csv(prompts_path)

    augmentation_path = 'cache/v2.11d.2/evaluate_all.pkl'
    with open(augmentation_path, 'rb') as f:
        augmentation_df = pickle.load(f)
    print(augmentation_df)


    dec_fact_index = get_dec_fact_index(prompts_df=prompts_df)
    for fid in dec_fact_index[:100]:
        print('===================================================')
        print(augmentation_df.loc[fid])
        prompts_by_fact = prompts_df.loc[prompts_df['fact_id'] == fid]
        print(prompts_by_fact, '\n')



    

if __name__ == '__main__':
    main()