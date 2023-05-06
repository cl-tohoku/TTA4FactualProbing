
import pandas as pd
import random

def write_confusion_md(md_path:str, tt, tf, ft, ff):
    with open(md_path, 'a') as f:
        f.write('## confuison matrix\n')
        f.write('||c_augment|w_augment|\n')
        f.write('|:-:|:-:|:-:|\n')
        f.write(f'|c_original|{tt}|{tf}|\n')
        f.write(f'|w_original|{ft}|{ff}|\n')
        f.write('\n')

def write_acc_md(md_path:str, acc_o, acc_a):
    with open(md_path, 'a') as f:
        f.write('## accuracy\n')
        f.write(f'original: {acc_o}  \n')
        f.write(f'augment: {acc_a}  \n')
        f.write(f'rel. dif.: {acc_a/acc_o}  \n')
        f.write('\n')


def write_rand_md(md_path:str, dir_path, ttidx, tfidx, ftidx, ffidx):
    n_samples = 3
    prompt_df = pd.read_csv(dir_path+'prompts.csv', index_col=0)
    with open(md_path, 'a') as f:
        f.write('## examples  \n')
        f.write('### TT  \n')
        for fid in random.choices(ttidx, k=n_samples):
            f.write('{}  \n\n'.format(prompt_df.loc[prompt_df['fact_id'] == fid].to_markdown()))
        f.write('### TF  \n')
        for fid in random.choices(tfidx, k=n_samples):
            f.write('{}  \n\n'.format(prompt_df.loc[prompt_df['fact_id'] == fid].to_markdown()))
        f.write('### FT  \n')
        for fid in random.choices(ftidx, k=n_samples):
            f.write('{}  \n\n'.format(prompt_df.loc[prompt_df['fact_id'] == fid].to_markdown()))
        f.write('### FF  \n')
        for fid in random.choices(ffidx, k=n_samples):
            f.write('{}  \n\n'.format(prompt_df.loc[prompt_df['fact_id'] == fid].to_markdown()))
    return 0

def folder2name(folder):
    dictionary = {
        'v2.11d.gpt3.1': 't5-small.gpt3',
        'v2.11d.gpt3.2': 't5-large.gpt3',
        'v2.11d.gpt3.3': 'flan-small.gpt3',
        'v2.11d.gpt3.4': 'flan-large.gpt3',
    }
    return dictionary[folder]

def main(folder):
    name = folder2name(folder)
    dir_path = f'cache/{folder}/'
    judge_df = pd.read_csv(dir_path+'augmentations.csv', index_col=0)
    report_path = f'analysis/md/{name}.md'
    with open(report_path, 'w') as f:
        f.write(f'# counts of {name}\n')
    print(judge_df.loc[(judge_df['c_original']) & (judge_df['c_all'])])
    ttidx = judge_df.loc[(judge_df['c_original']) & (judge_df['c_all'])].index
    tfidx = judge_df.loc[(judge_df['c_original']) & (~judge_df['c_all'])].index
    ftidx = judge_df.loc[(~judge_df['c_original']) & (judge_df['c_all'])].index
    ffidx = judge_df.loc[(~judge_df['c_original']) & (~judge_df['c_all'])].index
    tt = len(ttidx)
    tf = len(tfidx)
    ft = len(ftidx)
    ff = len(ffidx)
    total = tt + tf + ft + ff
    write_confusion_md(report_path, tt, tf, ft, ff)
    write_acc_md(report_path, (tt + tf)/total, (tt + ft)/total)
    write_rand_md(report_path, dir_path, ttidx, tfidx, ftidx, ffidx)
    return 0

if __name__ == '__main__':
    
    main(folder = 'v2.11d.gpt3.3')