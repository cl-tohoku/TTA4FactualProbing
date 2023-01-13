import pandas as pd
import pickle
import matplotlib.pyplot as plt



def get_variance_dict(idx, prompts_df):
    prompts_df = prompts_df.loc[prompts_df['fact_id'].isin(idx)]  

    variations = [len(prompts_df.loc[prompts_df['fact_id'] == id].groupby('result').count().index) for id in idx]
    vdict = {}
    for v in variations:
        if v in vdict.keys():
            vdict[v] += 1
        else:
            vdict[v] = 1

    return vdict


def main(folder):
    aug_path = 'cache/{}/augmentations.csv'.format(folder)
    prompt_path = 'cache/{}/prompts.csv'.format(folder)
    aug_df = pd.read_csv(aug_path)
    aug_df = aug_df[['c_original', 'c_all']]
    TT = aug_df.loc[(aug_df['c_original']==True) & (aug_df['c_all']==True) ]
    TF = aug_df.loc[(aug_df['c_original']==True) & (aug_df['c_all']==False) ]
    FT = aug_df.loc[(aug_df['c_original']==False) & (aug_df['c_all']==True) ]
    FF = aug_df.loc[(aug_df['c_original']==False) & (aug_df['c_all']==False) ]

    # augmentation_path = 'cache/{}/evaluate_all.pkl'.format(folder)
    # with open(augmentation_path, 'rb') as f:
    #     augmentation_df = pickle.load(f)
    prompts_df = pd.read_csv(prompt_path)
    prompts_df['is_correct'] = prompts_df['is_correct'] * 1.0

    tt_dict = get_variance_dict(TT.index, prompts_df)
    tf_dict = get_variance_dict(TF.index, prompts_df)
    ft_dict = get_variance_dict(FT.index, prompts_df)
    ff_dict = get_variance_dict(FF.index, prompts_df)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].bar(x=list(tt_dict.keys()), height=list(tt_dict.values()))
    axs[0, 0].set_title('TT') 
    axs[0, 0].set_xlim(1, 30)
    axs[0, 1].bar(x=list(tf_dict.keys()), height=list(tf_dict.values()))
    axs[0, 1].set_title('TF') 
    axs[0, 1].set_xlim(1, 30)
    axs[1, 0].bar(x=list(ft_dict.keys()), height=list(ft_dict.values()))
    axs[1, 0].set_title('FT') 
    axs[1, 0].set_xlim(1, 30)
    axs[1, 1].bar(x=list(ff_dict.keys()), height=list(ff_dict.values()))
    axs[1, 1].set_title('FF') 
    axs[1, 1].set_xlim(1, 30)
    pos = axs[1, 0].get_position()
    new_pos = [pos.x0, pos.y0-0.03, pos.width, pos.height]
    axs[1, 0].set_position(new_pos)
    pos = axs[1, 1].get_position()
    new_pos = [pos.x0, pos.y0-0.03, pos.width, pos.height]
    axs[1, 1].set_position(new_pos)
    plt.savefig('analysis/img/variations.png')
    
    # print('\tAug.T\tAug.F')
    # print('Orig.T\t{}\t{}'.format(TT.count()[0]/125, TF.count()[0]/125))
    # print('Orig.F\t{}\t{}'.format(FT.count()[0]/125, FF.count()[0]/125))

    

if __name__ == '__main__':
    # print('---t5-small---')
    # main('v2.11d.5')
    print('\n---t5-11b---')
    main('v2.11d.4')