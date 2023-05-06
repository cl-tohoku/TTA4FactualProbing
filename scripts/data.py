from __future__ import annotations
import texthero as hero
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import os
import pickle
from textattack.augmentation import Augmenter, WordNetAugmenter
from textattack.transformations import WordSwapEmbedding
from concurrent.futures import ThreadPoolExecutor
import logging

from models import get_transformer_model
from tools import split_list, init_logging
from gpt3 import GPT3

def aggregate(texts: list, scores:list, score_max=None, method = 'sum'):
    if score_max != None:
        assert isinstance(score_max, float), "score_max need to be a float"
    aggregated_result = {}
    for text, score in zip(texts, scores):
        if method == 'sum':
            aggregated_result[text] = aggregated_result.get(text, 0) + score
        elif method == 'count':
            aggregated_result[text] = aggregated_result.get(text, 0) + 1

    aggregated_result = OrderedDict(sorted(aggregated_result.items(), key=lambda x: x[1], reverse=True))

    text_list = list(aggregated_result.keys())
    score_list = list(aggregated_result.values())
    if score_max != None:
        score_list = list(score_list / score_list[0] * score_max)

    return {"texts": text_list, "scores": score_list}




class ExpResult():
    logger = logging.Logger
    facts: Facts
    prompts: Prompts
    generations: Generations
    settings: dict
    is_correct_df: pd.DataFrame
    def __init__(self, settings:dict):
        self.settings = settings
        dir_path = "cache/{}".format(self.settings["name"])
        # print(self.settings["block_size"])
        if os.path.exists(dir_path+"/info.log"):
            os.remove(dir_path+"/info.log")
        self.logger = init_logging("ExpResult", dir_path, "info.log")
        self.logger.info(self.settings["name"])
        if self.settings["is_saved"]:
            self.logger.info("loading facts")
            self.facts = Facts(setting_name=self.settings["name"])
            self.logger.info("loaded facts")
            
            self.logger.info("loading prompts")
            self.prompts = Prompts(setting_name=self.settings["name"])
            self.logger.info("loaded prompts")
            # print(self.settings["block_size"])
            self.logger.info("loading generations")
            self.generations = Generations(setting_name=self.settings["name"])
            self.logger.info("loaded generations")

            self.logger.info("loaded results")
        else:
            os.makedirs(dir_path, exist_ok=True)

            self.logger.info("loading facts")
            self.facts = Facts(setting_name=self.settings["name"], fact_table_path= self.settings["fact_csv"])
            self.logger.info("loaded facts")

            self.logger.info("augmenting prompts")
            self.prompts = Prompts(
                facts_df=self.facts.df,
                augmentation_methods=self.settings["augmentation_methods"]
            )
            self.logger.info("augmentation done")
            
            self.logger.info("generating")
            self.generations= Generations(
                setting_name=self.settings["name"],
                fact_df= self.facts.df,
                prompt_df=self.prompts.df,
                model_args=self.settings["lm"],
                generation_args=self.settings["generation_args"],
                block_size=self.settings["block_size"],
                split=self.settings["split"]
            )
            self.logger.info("generation done")

            self.logger.info("getting top answer for each prompt")
            self.prompts.evaluate(
                generations_df=self.generations.df,
                facts_df=self.facts.df
            )
            self.logger.info("getting top answer done")
            
            # self.facts.evaluate(
            #     generations_df=self.generations.df,
            #     prompts_df=self.prompts.df,
            #     setting_name=self.settings["name"],
            #     column_name="all",
            #     prompt_types_to_exclude = []
            # )
            # print("augment(fact) evaluation done")

            with open(dir_path + "/facts_df.pkl", "wb") as f:
                pickle.dump(self.facts.df, f)

            with open(dir_path + "/prompts_df.pkl", "wb") as f:
                pickle.dump(self.prompts.df, f)

            with open(dir_path + "/generations_df.pkl", "wb") as f:
                pickle.dump(self.generations.df, f)

    def aggregate(
        self,
        aggregation_types: list[dict]
    ):
        aggr_todo = []
        for aggregation_type in aggregation_types:
            filepath = "cache/{}/evaluate_{}.pkl".format(self.settings["name"], aggregation_type["name"])
            if os.path.exists(filepath):
                continue
            else:
                aggr_todo.append(aggregation_type)

        print(aggr_todo)
        # for aggregation_type in aggr_todo:
        #     self.aggregate_thread(aggregation_type)

        print("step1")

        # for aggregation_type in aggr_todo:
        #     self.aggregate_thread(aggregation_type)
        with ThreadPoolExecutor(thread_name_prefix="thread") as executor:
            for aggregation_type in aggr_todo:
                executor.submit(self.aggregate_thread, aggregation_type)

        executor = ThreadPoolExecutor(max_workers=64)
        for aggregation_type in aggr_todo:
            executor.submit(self.aggregate_thread, aggregation_type)
        executor.shutdown()

        
        print("step2")
        for aggregation_type in aggregation_types:
            filepath = "cache/{}/evaluate_{}.pkl".format(self.settings["name"], aggregation_type["name"])
            print(filepath)
            with open(filepath, "rb") as f:
                df = pickle.load(f)
            df = df[['{}'.format(aggregation_type['name']), 'c_{}'.format(aggregation_type['name'])]]
            self.facts.df = pd.concat([self.facts.df, df], axis=1)

    def aggregate_thread(self, args):
        self.logger.info("thread_{}".format(args["name"]))
        if not "weight_dict" in args.keys():
            args["weight_dict"] = None
        if not "rand" in args.keys():
            args["rand"] = None
            args["rand_state"] = None
        self.facts.evaluate(
            generations_df=self.generations.df,
            prompts_df=self.prompts.df,
            setting_name=self.settings["name"],
            column_name=args["name"],
            prompt_types_to_exclude=args["exclude"],
            weight_dict = args["weight_dict"],
            rand=args['rand']
        ) 
        self.logger.info("thread_{} done".format(args["name"]))
        return 0

    def ablation(self):
        self.prompts.df = self.facts.ablation(self.prompts.df, self.generations.df)


class CorrectTable():
    df: pd.DataFrame
    def __init__(
        self, 
        facts_df: pd.DataFrame,
        prompts: Prompts
    ):
        augment_df = facts_df.loc[:,["c_all"]].copy().rename(columns={"c_all": "augment"})
        # augment_df = facts_df.loc[:,["c_deved"]].copy().rename(columns={"c_d": "augment"})
        num_prompts = prompts.get_num_prompts()
        prompt_is_corrects = prompts.df.loc[:,"is_correct"].values.reshape([-1, num_prompts])
        assert len(augment_df.index) == prompt_is_corrects.shape[0]
        prompt_df = pd.DataFrame(prompt_is_corrects, columns=["p{}".format(i) for i in range(num_prompts)])
        is_correct_df = pd.concat([augment_df, prompt_df], axis=1)
        is_correct_df["count"] = is_correct_df.sum(axis=1)
        self.df = is_correct_df.loc[is_correct_df["augment"].notnull()]

    def compare_original_to_augment(self):
        analyzation = {
            "inc": self.df[(self.df["augment"] == True) & (self.df["p0"] == False)].shape[0],
            "dec": self.df[(self.df["augment"] == False) & (self.df["p0"] == True)].shape[0]
        }
        print(analyzation["inc"])
        return analyzation

    

# fact table and its controls
# columns: id,s,p,o,type,ss,os,prompt, + aggregation and correct/incorrects
class Facts():
    df: pd.DataFrame
    is_evaluated: bool
    setting_name: str

    def __init__(
        self,
        setting_name: str,
        fact_table_path = None,
    ):
        self.setting_name = setting_name
        filepath = "cache/{}/facts_df.pkl".format(self.setting_name)
        if os.path.exists(filepath):
            with open("cache/{}/facts_df.pkl".format(setting_name), "rb") as f:
                self.df = pickle.load(f)
            self.is_evaluated = True
        else:
            assert fact_table_path != None, "need fact_table_path"
            assert isinstance(fact_table_path, str)
            self.df = pd.read_csv(fact_table_path, index_col=0)
            self.is_evaluated = False

    def evaluate(
        self,
        prompts_df: pd.DataFrame,
        generations_df: pd.DataFrame,
        setting_name: str,
        column_name: str,
        prompt_types_to_exclude: list,
        weight_dict = None,
        rand = None,
        rand_state = None
    ):
        filename = "cache/{}/evaluate_{}.pkl".format(setting_name, column_name)
        mid_df = self._get_results(
            prompts_df=prompts_df,
            generations_df=generations_df,
            column_name=column_name,
            prompt_types_to_exclude=prompt_types_to_exclude,
            weight_dict = weight_dict,
            rand = rand,
            rand_state=rand_state
            )

        self._judge_correct(mid_df=mid_df ,prompts_df=prompts_df, generations_df=generations_df, column_name=column_name, filename = filename)

    def _get_results(
        self,
        prompts_df: pd.DataFrame,
        generations_df: pd.DataFrame,
        column_name: str,
        prompt_types_to_exclude: list,
        weight_dict = None,
        rand = None,
        rand_state = None
    ):
        if weight_dict == None:
            generations_df["weighted_score"] = generations_df["score"]
        else:
            generations_df["weighted_score"] = generations_df.apply(
                lambda row:
                    row["score"] * weight_dict[prompts_df.loc[row["prompt_id"]]["type"]],
                axis=1
            )
        prompt_gb_fact = prompts_df.groupby("fact_id")
        result_dict = {}
        score_dict = {}
        score_total_dict = {}
        gold_score = {}
        for fact_id, prompts in prompt_gb_fact:
            gold = self.df.loc[fact_id]['os']
            if rand == None:
                prompts = prompts.loc[~prompts["type"].isin(prompt_types_to_exclude)]
            else:
                prompts = pd.concat([prompts.loc[prompts['type'] == '0original'], prompts.loc[prompts['type'] != '0original'].sample(n=rand - 1, random_state=rand_state)])
            generation_df = generations_df.loc[generations_df["prompt_id"].isin(prompts.index)]
            if len(generation_df.index) != 0:
                texts = list(generation_df["generation"].values)
                scores = list(generation_df["weighted_score"].values)
                aggregation_result = aggregate(texts=texts, scores=scores)
                result_dict[fact_id] = aggregation_result["texts"][0]
                score_dict[fact_id] = aggregation_result["scores"][0]
                if gold in aggregation_result["texts"]:
                    gold_score[fact_id] = aggregation_result["scores"][aggregation_result["texts"].index(gold)]
                else:
                    gold_score[fact_id] = 0
                score_total_dict[fact_id] = sum(aggregation_result["scores"])
        result_df = pd.DataFrame({
            "fact_id":result_dict.keys(),
            column_name: result_dict.values(),
            'score': score_dict.values(),
            'gold_score': gold_score.values(),
            'score_total': score_total_dict.values(),
        }).set_index('fact_id')
        return pd.concat([self.df, result_df], axis=1)
    
    def _judge_correct(
        self,
        mid_df: pd.DataFrame,
        prompts_df: pd.DataFrame,
        generations_df: pd.DataFrame,
        column_name: str,
        filename: str
    ):
        valids = mid_df.loc[~mid_df[column_name].isnull(), ["os", column_name, 'score','gold_score', 'score_total']].copy()
        # valids["c_{}".format(column_name)] = valids.apply(lambda row: row[column_name].lower() == row["os"].lower(), axis=1)
        valids["c_{}".format(column_name)] = valids.apply(lambda row: row[column_name] == row["os"], axis=1)
        with open(filename, "wb") as f:
            df = valids
            # df = valids.loc[:, [column_name, "c_{}".format(column_name)]]
            pickle.dump(df, f)
        # self.df = pd.concat([self.df, valids["c_{}".format(column_name)]], axis=1)

    def get_corrects_table(self):
        df = pd.concat([self.df["p"],self.df.filter(regex="c\_.*")], axis=1)
        # print(df)
        df = df.loc[df["c_original"].notnull()]
        df.to_csv("cache/{}/augmentations.csv".format(self.setting_name))
        df.groupby("p").sum().T.to_csv("cache/{}/augmentation_summary.csv".format(self.setting_name), sep="\t")
        df.loc[:,["p","c_original"]].groupby("p").count().T.to_csv("cache/{}/valid_facts_per_pid.csv".format(self.setting_name), sep="\t")
        # n_correct_df = pd.DataFrame(df.sum())
        # n_correct_df.to_csv("cache/{}/augmentation_summary.csv".format(self.setting_name), sep="\t")
        # print(n_correct_df)
        # print(df.value_counts())
        # print(df.sum().sort_values())

    def get_num_evaluated_facts(self):
        df = self.df.loc[:,["p","c_all"]]
        df = df.loc[df["c_all"].notnull()] # remove null rows
        # print(df)
        return len(df.index)
    
    def get_pids(self):
        ndarray = self.df["p"].unique()
        return list(ndarray)

    def ablation(self, prompts_df, generations_df):
        prompt_gb_fact = prompts_df.groupby("fact_id")
        top_confidence = {}
        gold_confidence = {}
        score_total_dict = {}
        for fact_id, prompts in prompt_gb_fact:
            for i in range(len(prompts.index)):
                print(prompts.index[i])
                removed_index = [prompts.index[idx] for idx in range(len(prompts.index)) if idx != i]
                generation_df = generations_df.loc[generations_df["prompt_id"].isin(removed_index)]
                if len(generation_df.index) != 0:
                    texts = list(generation_df["generation"].values)
                    scores = list(generation_df["score"].values)
                    aggregation_result = aggregate(texts=texts, scores=scores)
                    score_total = sum(aggregation_result["scores"])
                    gold = self.df.loc[fact_id]['os']
                    if gold in aggregation_result["texts"]:
                        gold_confidence[prompts.index[i]] = aggregation_result["scores"][aggregation_result["texts"].index(gold)] / score_total
                    else:
                        gold_confidence[prompts.index[i]] = 0
                    top_confidence[prompts.index[i]] = aggregation_result["scores"][0] / score_total

        result_df = pd.DataFrame({
            "prompt_id":gold_confidence.keys(),
            "gold_conf": gold_confidence.values(),
            'top_conf': top_confidence.values(),
        }).set_index('prompt_id')
        print(pd.concat([prompts_df, result_df], axis=1))
        return pd.concat([prompts_df, result_df], axis=1)

    # def prompt_effect(self, prompt_df):
        



class Prompts():
    df: pd.DataFrame
    is_evaluated: bool
    def __init__(
        self,
        setting_name = None,
        facts_df = None,
        augmentation_methods = None
    ):
        if setting_name != None:
            assert isinstance(setting_name, str)
            with open("cache/{}/prompts_df.pkl".format(setting_name), "rb") as f:
                self.df = pickle.load(f)
            self.is_evaluated = True
        else:
            assert isinstance(facts_df, pd.DataFrame)
            assert isinstance(augmentation_methods, list)   # list[dict]
            self.df = facts_df[["prompt"]].copy()
            self.df["fact_id"] = self.df.index
            self.df["type"] = "0original"
            self.df["score"] = 1.0
            self._augment(augmentation_methods, facts_df)
            self.is_evaluated = False

    def _augment(self, augmentation_methods: list[dict], facts_df):
        fact_ids = self.df["fact_id"]
        method_dict = {
            "backtranslation": self._backtranslate,
            "textattack": self._textattack,
            "heroclean": self._hero_clean,
            "gpt3": self._gpt3,
            "template": self._template
        }
        dfs = []
        for method in augmentation_methods:
            print(method["name"])
            if os.path.exists(method["filepath"]):
                with open(method["filepath"], "rb") as f:
                    df = pickle.load(f)
            else:
                if method["name"] == "template":
                    method['args']['facts_df'] = facts_df
                augment_result = method_dict[method["name"]](**method["args"])
                df = pd.DataFrame({
                    "fact_id": fact_ids.repeat(method["args"]["num_return_sequences"]),
                    "prompt": augment_result["texts"],
                    "score": augment_result["scores"],
                    "type": method["label"]
                })
                with open(method["filepath"], "wb") as f:
                    pickle.dump(df, f)
            dfs.append(df)
        self.df = pd.concat([self.df]+dfs, axis=0).sort_values(["fact_id", "type", "score"], ascending=[True, True, False]).reset_index(drop=True)

    def _gpt3(self, num_return_sequences:int):
        logger = init_logging("gpt3", "cache", filename="gpt3.log")
        original_prompts = self.df.loc[self.df["type"] == "0original", "prompt"]
        gpt3 = GPT3()
        augmented_prompts = []
        augmented_scores = []
        for fid, original_prompt in enumerate(original_prompts.values.tolist()):
            last = 12499
            if fid < last:
                continue
            if fid == last:
                with open("gpt3.tmp.pkl", "rb") as f:
                    result = pickle.load(f)
                    augmented_prompts = result["texts"]
                    augmented_scores = [1]*len(augmented_prompts)
                continue

            augment_result = gpt3.get_paraphrases(original_sentence=original_prompt)
            augment_score = [1]*len(augment_result)
            if len(augment_result) < num_return_sequences:
                n_more = num_return_sequences - len(augment_result)
                logger.warning("fid:{}\t{} paraphrases".format(fid, len(augment_result)))
                augment_result += [""]*n_more
                augment_score += [0]*n_more
            elif len(augment_result) > num_return_sequences:
                logger.warning("fid:{}\t{} paraphrases".format(fid, len(augment_result)))
                augment_result = augment_result[:10]
                augment_score += augment_score[:10]
            augmented_prompts += augment_result
            augmented_scores += augment_score
            logger.info("fid:{}\t".format(fid))
            with open("gpt3.tmp.pkl", "wb") as f:
                pickle.dump({"texts": augmented_prompts, "scores": augmented_scores}, f)
        return {"texts": augmented_prompts, "scores": augmented_scores}

    def _template(self, template_df_path: str, facts_df: pd.DataFrame, num_return_sequences: int):
        template_df = pd.read_csv(template_df_path)
        augmented_prompts = []
        augmented_scores = []
        print(facts_df)
        for idx, row in facts_df.iterrows():
            templates = template_df.loc[template_df['pid'] == row['p']]['template'].values.tolist()
            print(templates)
            augmented_prompts += [template.replace('[MASK]',row['ss']) for template in templates]
            augmented_scores += [1]*len(templates)

        return {"texts": augmented_prompts, "scores": augmented_scores}
        

    def _textattack(
        self,
        num_return_sequences: int,
        augmenter_type: str
    ):
        augmenter = self._get_augmenter(augment_type=augmenter_type)
        augmenter.transformations_per_example = num_return_sequences
        original_prompts = self.df.loc[self.df["type"] == "0original", "prompt"]
        augmented_prompts = []
        augmented_scores = []
        for original_prompt in original_prompts.values.tolist():
            augment_result = augmenter.augment(original_prompt)
            augmented_scores += [1]*len(augment_result)
            if len(augment_result) < num_return_sequences:
                n_more = num_return_sequences - len(augment_result)
                augment_result += [""]*n_more
                augmented_scores+= [0]*n_more
            augmented_prompts += augment_result
        return {"texts": augmented_prompts, "scores": augmented_scores}
    
    def _hero_clean(
        self,
        num_return_sequences: int
    ):
        original_prompts = self.df.loc[self.df["type"] == "0original", "prompt"]
        keys = hero.remove_stopwords(original_prompts)
        keys = hero.remove_diacritics(keys)
        return {"texts": keys, "scores": [1.0]*len(keys)}

    def _get_augmenter(
        self,
        augment_type: str
    ) -> Augmenter:
        if augment_type == "wordnet":
            return WordNetAugmenter(pct_words_to_swap=0.2)
        elif augment_type == "wordswap":
            return Augmenter(transformation = WordSwapEmbedding())

    def _backtranslate(
        self,
        tar_lang: str,
        num_return_sequences: int
    ):
        assert num_return_sequences > 0, "num_return_sequences need to be positive"
        assert num_return_sequences <= 64, "num_return_sequences need to be less than 100" 
        lm_src2tar = {
            "lm": "marianmt",
            "label": "en-{}".format(tar_lang),
            "pretrained_model_name": "Helsinki-NLP/opus-mt-en-{}".format(tar_lang),
            "map_name":None
        }
        lm_tar2src = {
            "lm": "marianmt",
            "label": "{}-en".format(tar_lang),
            "pretrained_model_name": "Helsinki-NLP/opus-mt-{}-en".format(tar_lang),
            "map_name":None
        }
        sequence_per_transform = num_return_sequences * 2
        beam_search_args = {
            "do_sample": False, # do greedy or greedy beam-search
            "output_scores": True,
            "return_dict_in_generate": True,
            "num_beams":  sequence_per_transform, # beam-search if >2
            "num_return_sequences": sequence_per_transform, # need to be <= num_beams
            "bad_words_ids": [[54]], # "?":54, "\", "\u2581?": 99 in en-fr
            "max_new_tokens": 100,
        }
        original_prompts = self.df.loc[self.df["type"] == "0original", "prompt"]
        
        ## src to tar translation
        tar_result = self._translate(
            original_prompts.values.tolist(),
            model_args=lm_src2tar,
            generation_args=beam_search_args
        )

        ## tar to src translation
        back_result = self._translate(
            tar_result["texts"],
            model_args=lm_tar2src,
            generation_args=beam_search_args
        )

        ## aggregate backtranslation
        final_texts = []
        final_scores = [] 
        back_result_per_fact_text = split_list(back_result["texts"], num_cols=sequence_per_transform ** 2)
        back_result_per_fact_score = np.reshape(back_result["scores"], [-1, sequence_per_transform])
        back_result_per_fact_score = np.reshape((back_result_per_fact_score.T * tar_result["scores"]).T, [-1, sequence_per_transform**2])
        for i in range(len(back_result_per_fact_text)):
            aggregated_backtranslation = aggregate(
                texts = back_result_per_fact_text[i],
                scores=back_result_per_fact_score[i],
                score_max= 1.0
            )
            if len(aggregated_backtranslation["texts"]) < num_return_sequences:
                n_more = num_return_sequences - len(aggregated_backtranslation["texts"])
                aggregated_backtranslation["texts"] += [""]*n_more
                aggregated_backtranslation["scores"] += [0.0]*n_more
            final_texts += aggregated_backtranslation["texts"][:num_return_sequences]
            final_scores += aggregated_backtranslation["scores"][:num_return_sequences]
        
        return {"texts": final_texts, "scores": final_scores}

    def _translate(
        self,
        texts: list[str],
        model_args: dict,
        generation_args: dict,
    ):
        result = {"texts": [], "scores": np.empty(0)}
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model, tokenizer = get_transformer_model(model_args)
        prompt_tensors = tokenizer(
            texts,
            return_tensors="pt",
            padding=True
        )
        
        block_size = 16
        input_id_blocks = split_list(sequence=prompt_tensors.input_ids, num_cols=block_size)
        attention_mask_blocks = split_list(sequence=prompt_tensors.attention_mask, num_cols=block_size)
        generated_tensors = []
        generation_scores = []
        pad_idx = tokenizer("<pad>").input_ids[0]
        print(pad_idx)
        with torch.no_grad():
            for bid, (input_ids, attention_mask) in enumerate(zip(input_id_blocks, attention_mask_blocks)):
                output = model.generate(
                    input_ids = input_ids.to(device),
                    attention_mask = attention_mask.to(device),
                    **generation_args
                )
                generated_tensors.append(output.sequences)
                generation_scores.append(output.sequences_scores)
                if bid == 1:
                    print(generated_tensors)
            generated_tensors = [F.pad(t, (0, generation_args["max_new_tokens"] - t.size()[-1] + 1), "constant", pad_idx) for t in generated_tensors]
            generated_tensors = torch.cat(generated_tensors).to("cpu").detach()
            generation_scores = torch.cat(generation_scores).to("cpu").detach()
            generation_scores = torch.exp(generation_scores).to('cpu').detach().numpy().copy()
            generated_texts = tokenizer.batch_decode(generated_tensors, skip_special_tokens=True)
            # for block in text_blocks:
            #     inputs = tokenizer(
            #         block,
            #         return_tensors="pt",
            #         padding=True
            #     ).to(device)
            #     outputs =model.generate(
            #         input_ids = inputs.input_ids,
            #         attention_mask = inputs.attention_mask,
            #         **generation_args
            #     )
            #     result["texts"] += tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            #     result["scores"] = np.concatenate([result["scores"], torch.exp(outputs.sequences_scores).to('cpu').detach().numpy().copy()], 0)
            result = {"texts": generated_texts, "scores": generation_scores}
        return result

    def evaluate(
        self,
        generations_df: pd.DataFrame,
        facts_df: pd.DataFrame
    ):
        assert not self.is_evaluated
        self._get_results(generations_df=generations_df)
        self._judge_correct(generations_df=generations_df, facts_df=facts_df)
        self.is_evaluated = True

    def _get_results(
        self,
        generations_df: pd.DataFrame,
    ):
        generation_gb_prompt = generations_df.groupby("prompt_id")
        result_dict = {}
        for prompt_id, generations in generation_gb_prompt:
            result_dict[prompt_id] = generations.sort_values("score", ascending=False).iloc[0]["generation"]
        result_df = pd.DataFrame({
            "prompt_id":result_dict.keys(),
            "result": result_dict.values()
        }).set_index('prompt_id')
        self.df = pd.concat([self.df, result_df], axis=1)
    
    def _judge_correct(
        self,
        generations_df: pd.DataFrame,
        facts_df: pd.DataFrame,
    ):
        valids = self.df.loc[~self.df["result"].isnull(), ["fact_id", "result"]].copy()
        valids["is_correct"] = valids.apply(lambda row: row["result"] == facts_df.loc[row["fact_id"]]["os"] if row["result"] != np.nan else np.nan, axis=1)
        self.df = pd.concat([self.df, valids["is_correct"]], axis=1)

    def get_num_prompts(self):
        prompt_count_sr = self.df.groupby("fact_id").count()["prompt"]
        assert prompt_count_sr.var() == 0.0
        return prompt_count_sr[0].item()

    def get_corrects(self):
        assert self.is_evaluated
        return self.df.loc[self.df["is_correct"] == True, :]


class Generations():

    def __init__(
        self,
        setting_name = None,
        fact_df = None,
        prompt_df= None,
        model_args= None,
        generation_args= None,
        block_size= None,
        split = None, # {"path":filepath, "slice":[0, 100]}
    ):
        filepath = "cache/{}/generations_df.pkl".format(setting_name)
        if split != None:
            filepath = "cache/{}/generations_{}_df.pkl".format(setting_name, split["path"])
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                self.df = pickle.load(f)
        else:
            assert isinstance(fact_df, pd.DataFrame)
            assert isinstance(prompt_df, pd.DataFrame)
            assert isinstance(model_args, dict)
            assert isinstance(generation_args, dict)

            logger = init_logging("generation", "cache/{}".format(setting_name), filename="info.log")
            logger.info(block_size)
            model, tokenizer = get_transformer_model(model_args)
            logger.info(model.hf_device_map)    

            ## remove prompts with o longer than n_token to generate
            df = fact_df.loc[:, ["os"]]
            df["len"] = df.apply(lambda row: len(tokenizer(row["os"], return_tensors="pt").input_ids[0]), axis=1)
            valid_fact_ids = df.loc[df["len"] <= generation_args["max_new_tokens"]].index
            prompt_ids = prompt_df.loc[prompt_df["fact_id"].isin(valid_fact_ids)].index

            if split != None:
                if split["slice"][0] != None and split["slice"][1] != None:
                    prompt_ids = prompt_ids[split["slice"][0]:split["slice"][1]]
                elif split["slice"][0] != None and split["slice"][1] == None:
                    prompt_ids = prompt_ids[split["slice"][0]:]
            prompts =prompt_df.loc[prompt_ids.to_list(), "prompt"].to_list()
            if(model_args["label"] in ["gptjt"]):
                prompts = [prompt.replace("<br>", "\n") for prompt in prompts]
            print(prompts)

            if(setting_name in ["test-t03b"]):
                generation_args["max_new_tokens"] == None
            tensor_path = "cache/{}/prompt_tensors.pkl".format(setting_name)


            

            if(model_args["label"] in ["gptjt"]):
                model.eval()
                generated_texts = []
                generation_scores = []
                for prompt in prompts:
                    inputs = tokenizer(
                        prompts,
                        return_tensors="pt"
                    )
                    output = model.generate(
                        input_ids = inputs.input_ids.to('cuda'),
                        **generation_args
                    )
                    generated_texts += tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
                    generation_scores.append(output.sequences_scores)
                generation_scores = torch.cat(generation_scores).to("cpu").detach()
                generation_scores = torch.exp(generation_scores).to('cpu').detach().numpy().copy()
            else:
                if split != None:
                    tensor_path = "cache/{}/prompt_tensors_{}_df.pkl".format(setting_name, split["path"])
                if os.path.exists(tensor_path):
                    with open(tensor_path, "rb") as f:
                        prompt_tensors = pickle.load(f)
                else:
                    prompt_tensors = tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True
                    )
                    with open(tensor_path, "wb") as f:
                        pickle.dump(prompt_tensors,f)
                    

                input_id_blocks = split_list(sequence=prompt_tensors.input_ids, num_cols=block_size)
                attention_mask_blocks = split_list(sequence=prompt_tensors.attention_mask, num_cols=block_size)

                model.eval()
                generated_tensors = []
                generation_scores = []

                with torch.no_grad():
                    for bid, (input_ids, attention_mask) in enumerate(zip(input_id_blocks, attention_mask_blocks)):
                        logger.info(bid)
                        output = model.generate(
                            input_ids = input_ids.to('cuda'),
                            attention_mask = attention_mask.to('cuda'),
                            **generation_args
                        )
                        generated_tensors.append(output.sequences)
                        generation_scores.append(output.sequences_scores)

                pad_idx = tokenizer("<pad>").input_ids[0]
                generated_tensors = [F.pad(t, (0, generation_args["max_new_tokens"] - t.size()[-1] + 1), "constant", pad_idx) for t in generated_tensors]
                generated_tensors = torch.cat(generated_tensors).to("cpu").detach()
                generation_scores = torch.cat(generation_scores).to("cpu").detach()
                generation_scores = torch.exp(generation_scores).to('cpu').detach().numpy().copy()
                generated_texts = tokenizer.batch_decode(generated_tensors, skip_special_tokens=True)
            df = pd.DataFrame({
                "prompt_id": prompt_ids.repeat(generation_args["num_return_sequences"]),
                "generation": generated_texts,
                "score": generation_scores,
            })
            print(df)
            with open(filepath, "wb") as f:
                pickle.dump(df, f)
            # with open("cache/btfr4_gen.pkl", "rb") as f:
            #     df = pickle.load(f)
            self.df = df