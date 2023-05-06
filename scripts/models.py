import transformers
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration



from device_map import get_map


def t5(pretrained_model_name, map_name = None):
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    if map_name == None:
        device_map = "auto"
    else:
        device_map = get_map(map_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name, device_map=device_map)
    return model, tokenizer

def marianmt(pretrained_model_name, map_name = None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = MarianTokenizer.from_pretrained(pretrained_model_name)
    model = MarianMTModel.from_pretrained(pretrained_model_name).to(device).eval()
    return model, tokenizer


# def gptjt(pretrained_model_name, map_name = None):
#     tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
#     tokenizer.pad_token = tokenizer.eos_token
#     if map_name == None:
#         device_map = "auto"
#     else:
#         device_map = get_map(map_name)
#     model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, device_map=device_map)
#     return model, tokenizer

def flan(pretrained_model_name, map_name = None):
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name)
    if map_name == None:
        device_map = "auto"
    else:
        device_map = get_map(map_name)
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name, device_map=device_map)
    return model, tokenizer


def get_transformer_model(param):
    get_model = dict(
        t5 = t5,
        marianmt = marianmt,
        # gptjt = gptjt,
        t0 = t5,
        flan = flan
    )
    return get_model[param["lm"]](param["pretrained_model_name"], param["map_name"])

if __name__ == '__main__':
    param = {
        "lm": "t5",
        "label": "t5-large",
        "pretrained_model_name": "google/t5-large-ssm-nq",
        "map_name": "auto",
    }
    beam_search_args = {
            "do_sample": False, # do greedy or greedy beam-search
            "output_scores": True,
            "return_dict_in_generate": True,
            "num_beams":  10, # beam-search if >2
            "num_return_sequences": 10, # need to be <= num_beams
            # "bad_words_ids": [[54]], # "?":54, "\", "\u2581?": 99 in en-fr
            "max_new_tokens": 100,
    }
    model, tokenizer = get_transformer_model(param)
    prompts = ['Where is the capital of Alaska?']
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True
    )
    model.eval()
    with torch.no_grad():
        output = model.generate(
            input_ids = inputs['input_ids'].to('cuda'),
            attention_mask = inputs['attention_mask'].to('cuda'),
            **beam_search_args
        )
    generated_texts = tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)
    print(generated_texts)
    
    