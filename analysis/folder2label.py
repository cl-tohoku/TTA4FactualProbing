def folder2label(folder):
    dictionary = {
        'v2.11d.5': 'T5-small',
        'v2.11d.5.cnt': 'T5-small (count)',
        'v2.11d.2': 'T5-large',
        'v2.11d.2.cnt': 'T5-large (count)',
        'v2.11d.3': 'T5-3B',
        'v2.11d.3.cnt': 'T5-3B (count)',
        'v2.11d.4': 'T5-11B',
        'v2.11d.4.cnt': 'T5-11B (count)',
        'v2.11d.flan.uncased': 'Flan-xl',
        'v2.11d.flan-small.uncased': 'Flan-small',
        'v2.11d.t03b': 'T0-3B',
        'v2.11d.gpt3.1': 'T5-small (GPT3)',
        'v2.11d.gpt3.2': 'T5-large (GPT3)',
        'v2.11d.gpt3.3': 'FLAN-small (GPT3)',
        'v2.11d.gpt3.4': 'FLAN-xl (GPT3)',
        'v2.11d.gpt3.5': 'T5-3B (GPT3)',
        'v2.11d.gpt3.6': 'T5-11B (GPT3)',
    }
    return dictionary[folder]