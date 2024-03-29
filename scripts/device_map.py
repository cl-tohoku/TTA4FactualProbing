maps = {
    "auto":"auto",
    "t5-b-8":
        {
            "shared": 0,
            "lm_head": 1,
            "encoder.embed_tokens": 1,
            "encoder.block.0": 1,
            "encoder.block.1": 1,
            "encoder.block.2": 1,
            "encoder.block.3": 1,
            "encoder.block.4": 1,
            "encoder.block.5": 1,
            "encoder.block.6": 1,
            "encoder.block.7": 1,
            "encoder.block.8": 1,
            "encoder.block.9": 1,
            "encoder.block.10": 1,
            "encoder.block.11": 1,
            "encoder.block.12": 1,
            "encoder.block.13": 2,
            "encoder.block.14": 2,
            "encoder.block.15": 2,
            "encoder.block.16": 2,
            "encoder.block.17": 2,
            "encoder.block.18": 2,
            "encoder.block.19": 2,
            "encoder.block.20": 2,
            "encoder.block.21": 2,
            "encoder.block.22": 2,
            "encoder.block.23": 2,
            "encoder.final_layer_norm": 2,
            "encoder.dropout": 2,
            "decoder.embed_tokens": 2,
            "decoder.block.0": 3,
            "decoder.block.1": 3,
            "decoder.block.2": 3,
            "decoder.block.3": 3,
            "decoder.block.4": 3,
            "decoder.block.5": 3,
            "decoder.block.6": 4,
            "decoder.block.7": 4,
            "decoder.block.8": 4,
            "decoder.block.9": 4,
            "decoder.block.10": 4,
            "decoder.block.11": 4,
            "decoder.block.12": 5,
            "decoder.block.13": 5,
            "decoder.block.14": 5,
            "decoder.block.15": 5,
            "decoder.block.16": 5,
            "decoder.block.17": 5,
            "decoder.block.18": 6,
            "decoder.block.19": 6,
            "decoder.block.20": 6,
            "decoder.block.21": 6,
            "decoder.block.22": 6,
            "decoder.block.23": 6,
            "decoder.final_layer_norm": 7,
            "decoder.dropout": 7
        },
    "t5-small-2":
        {
            "shared": 0,
            "decoder.embed_tokens": 0,
            "encoder.embed_tokens": 0,
            "encoder.block.0": 0,
            "encoder.block.1": 0,
            "encoder.block.2": 0,
            "encoder.block.3": 0,
            "encoder.block.4": 0,
            "encoder.block.5": 1,
            "encoder.block.6": 1,
            "encoder.block.7": 1,
            "encoder.final_layer_norm": 1,
            "encoder.dropout": 1,
            "decoder.block": 1,
            "decoder.final_layer_norm": 1,
            "decoder.dropout": 1,
            "lm_head": 1
        },
    "t5-large-Glarge":
        {
            'shared': 0,
            'lm_head': 1,
            'encoder.embed_tokens': 1,
            'encoder.block.0': 1,
            'encoder.block.1': 1,
            'encoder.block.2': 1,
            'encoder.block.3': 1,
            'encoder.block.4': 1,
            'encoder.block.5': 1,
            'encoder.block.6': 1,
            'encoder.block.7': 1,
            'encoder.block.8': 1,
            'encoder.block.9': 1,
            'encoder.block.10': 1,
            'encoder.block.11': 1,
            'encoder.block.12': 1,
            'encoder.block.13': 1,
            'encoder.block.14': 1,
            'encoder.block.15': 1,
            'encoder.block.16': 2,
            'encoder.block.17': 2,
            'encoder.block.18': 2,
            'encoder.block.19': 2,
            'encoder.block.20': 2,
            'encoder.block.21': 2,
            'encoder.block.22': 2,
            'encoder.block.23': 2,
            'encoder.final_layer_norm': 2,
            'encoder.dropout': 2,
            'decoder.embed_tokens': 2,
            'decoder.block.0': 2,
            'decoder.block.1': 2,
            'decoder.block.2': 2,
            'decoder.block.3': 2,
            'decoder.block.4': 2,
            'decoder.block.5': 2,
            'decoder.block.6': 2,
            'decoder.block.7': 3,
            'decoder.block.8': 3,
            'decoder.block.9': 3,
            'decoder.block.10': 3,
            'decoder.block.11': 2,
            'decoder.block.12': 3,
            'decoder.block.13': 3,
            'decoder.block.14': 3,
            'decoder.block.15': 3,
            'decoder.block.16': 3,
            'decoder.block.17': 3,
            'decoder.block.18': 3,
            'decoder.block.19': 3,
            'decoder.block.20': 3,
            'decoder.block.21': 3,
            'decoder.block.22': 3,
            'decoder.block.23': 3,
            'decoder.final_layer_norm': 3,
            'decoder.dropout': 3
        },
    "flan-4":{'shared': 0,
            'decoder.embed_tokens': 1,
            'encoder.embed_tokens': 1,
            'encoder.block.0': 1,
            'encoder.block.1': 1,
            'encoder.block.2': 1,
            'encoder.block.3': 1,
            'encoder.block.4': 1,
            'encoder.block.5': 1,
            'encoder.block.6': 1,
            'encoder.block.7': 1,
            'encoder.block.8': 1,
            'encoder.block.9': 1,
            'encoder.block.10': 1,
            'encoder.block.11': 1,
            'encoder.block.12': 1,
            'encoder.block.13': 1,
            'encoder.block.14': 1,
            'encoder.block.15': 1,
            'encoder.block.16': 1,
            'encoder.block.17': 1,
            'encoder.block.18': 1,
            'encoder.block.19': 2,
            'encoder.block.20': 2,
            'encoder.block.21': 2,
            'encoder.block.22': 2,
            'encoder.block.23': 2,
            'encoder.final_layer_norm': 2,
            'encoder.dropout': 2,
            'decoder.block.0': 2,
            'decoder.block.1': 2,
            'decoder.block.2': 2,
            'decoder.block.3': 2,
            'decoder.block.4': 2,
            'decoder.block.5': 2,
            'decoder.block.6': 2,
            'decoder.block.7': 2,
            'decoder.block.8': 2,
            'decoder.block.9': 3,
            'decoder.block.10': 3,
            'decoder.block.11': 3,
            'decoder.block.12': 3,
            'decoder.block.13': 3,
            'decoder.block.14': 3,
            'decoder.block.15': 3,
            'decoder.block.16': 3,
            'decoder.block.17': 3,
            'decoder.block.18': 3,
            'decoder.block.19': 3,
            'decoder.block.20': 3,
            'decoder.block.21': 3,
            'decoder.block.22': 3,
            'decoder.block.23': 3,
            'decoder.final_layer_norm': 3,
            'decoder.dropout': 3,
            'lm_head': 3
        }
}

def get_map(name):
    return maps[name]