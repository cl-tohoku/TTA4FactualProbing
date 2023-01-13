import random
import pickle
import pandas as pd
import os

relations = [
    'P937',
    'P1365',
    'P50',
    'P119',
    'P69',
    'P27',
    'P20',
    'P156',
    'P641',
    'P740',
    'P155',
    'P19',
    'P1376',
    'P140',
    'P1366',
    'P159',
    'P131',
    'P495',
    'P103',
    'P407',
    'P36',
    'P37',
    'P1412',
    'P17',
    'P30',
]

files = [
    {
        'src': 'cache/v2.11d.5',
        'dev': 'cache/t5-small.dev',
        'test': 'cache/t5-small.test',
    },
    {
        'src': 'cache/v2.11d.2',
        'dev': 'cache/t5-large.dev',
        'test': 'cache/t5-large.test',
    },
    {
        'src': 'cache/v2.11d.3',
        'dev': 'cache/t5-3b.dev',
        'test': 'cache/t5-3b.test',
    },
    {
        'src': 'cache/v2.11d.4',
        'dev': 'cache/t5-11b.dev',
        'test': 'cache/t5-11b.test',
    },
    {
        'src': 'cache/v2.11d.flan',
        'dev': 'cache/flan-3b.dev',
        'test': 'cache/flan-3b.test',
    },
    {
        'src': 'cache/v2.11d.flan-small.uncased',
        'dev': 'cache/flan-small',
        'test': 'cache/flan-small.test',
    },
    {
        'src': 'cache/v2.11d.t03b.uncased',
        'dev': 'cache/t03b',
        'test': 'cache/t03b.test',
    },
]

dev = random.sample(relations, 15)
test = list(set(relations) - set(dev))

for models in files:
    os.makedirs(models['dev'], exist_ok=True)
    # facts
    with open(models['src']+'/facts_df.pkl', 'rb') as f:
        facts = pickle.load(f)
    facts = facts.loc[facts['p'].isin(dev)]
    with open(models['dev']+'/facts_df.pkl', 'wb') as f:
        pickle.dump(facts, f)
    fact_ids = facts.index

    # prompts
    with open(models['src']+'/prompts_df.pkl', 'rb') as f:
        prompts = pickle.load(f)
    prompts = prompts.loc[prompts['fact_id'].isin(fact_ids)]
    with open(models['dev']+'/prompts_df.pkl', 'wb') as f:
        pickle.dump(prompts, f)
    prompt_ids = prompts.index

    # generations
    with open(models['src']+'/generations_df.pkl', 'rb') as f:
        generations = pickle.load(f)
    generations = generations.loc[generations['prompt_id'].isin(prompt_ids)]
    with open(models['dev']+'/generations_df.pkl', 'wb') as f:
        pickle.dump(generations, f)


    os.makedirs(models['test'], exist_ok=True)
    # facts
    with open(models['src']+'/facts_df.pkl', 'rb') as f:
        facts = pickle.load(f)
    facts = facts.loc[facts['p'].isin(test)]
    with open(models['test']+'/facts_df.pkl', 'wb') as f:
        pickle.dump(facts, f)
    fact_ids = facts.index

    # prompts
    with open(models['src']+'/prompts_df.pkl', 'rb') as f:
        prompts = pickle.load(f)
    prompts = prompts.loc[prompts['fact_id'].isin(fact_ids)]
    with open(models['test']+'/prompts_df.pkl', 'wb') as f:
        pickle.dump(prompts, f)
    prompt_ids = prompts.index

    # generations
    with open(models['src']+'/generations_df.pkl', 'rb') as f:
        generations = pickle.load(f)
    generations = generations.loc[generations['prompt_id'].isin(prompt_ids)]
    with open(models['test']+'/generations_df.pkl', 'wb') as f:
        pickle.dump(generations, f)
