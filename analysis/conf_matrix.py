import pandas as pd

def main(folder):
    aug_path = 'cache/{}/augmentations.csv'.format(folder)
    prompt_path = 'cache/{}/prompts.csv'.format(folder)
    aug_df = pd.read_csv(aug_path)
    aug_df = aug_df[['c_original', 'c_all']]
    TT = aug_df.loc[(aug_df['c_original']==True) & (aug_df['c_all']==True) ]
    TF = aug_df.loc[(aug_df['c_original']==True) & (aug_df['c_all']==False) ]
    FT = aug_df.loc[(aug_df['c_original']==False) & (aug_df['c_all']==True) ]
    FF = aug_df.loc[(aug_df['c_original']==False) & (aug_df['c_all']==False) ]

    
    print('\tAug.T\tAug.F')
    print('Orig.T\t{}\t{}'.format(TT.count()[0]/125, TF.count()[0]/125))
    print('Orig.F\t{}\t{}'.format(FT.count()[0]/125, FF.count()[0]/125))

    

if __name__ == '__main__':
    print('---t5-small---')
    main('v2.11d.5')
    print('\n---t5-11b---')
    main('v2.11d.4')