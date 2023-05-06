import openai
import os
import sys
import time
import re
import pandas as pd
class GPT3():
    def __init__(self):
        self.model = "text-davinci-003" # GPT-3 engine here
        self.max_tokens=1024
        self.temperature=0.8
        self.top_p=1
        self.frequency_penalty=0
        self.presence_penalty=0
        openai.api_key = os.environ['CHATGPT_API']

    def gpt3_completion(self, prompt):
        # call GPT-3 API until result is provided and then return it
        response = None
        received = False
        # prevent over 600 requests per minute

        while not received:
            try:
                response = openai.Completion.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty
                )
                received = True
            except:
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                print("API error:", error)
                time.sleep(1)
        return response

    def _str2list(self, str):
        pattern = re.compile(r'^[0-9]+.(.+)')
        prompts = str.strip().split('\n')
        prompts = [pattern.match(prompt).group(1).strip() for prompt in prompts]
        return prompts

    
    def get_paraphrases(self, original_sentence):
        prompt = "Would you provide 10 paraphrases for the following question?\n{}".format(original_sentence)
        response = self.gpt3_completion(prompt)["choices"][0]["text"]
        print(response)
        paraphrases = self._str2list(response)

        return paraphrases





### Let's look at a simple QA for GPT3.

def templates():

    chatgpt = GPT3()
    paraphrased_templates = pd.DataFrame()
    for p, template in zip(pids, templates):
        prompt = "Would you provide 10 paraphrases for the following question?\n{}".format(template)
        response = chatgpt.gpt3_completion(prompt)["choices"][0]["text"]
        prompts = str2list(response)
        assert len(prompts) == 10, 'number of paraphrases not equal to 10'
        paraphrased_templates = pd.concat([paraphrased_templates, pd.DataFrame(list(zip([p]*10,  prompts)), columns=['pid', 'template'])], axis=0)
        print(pd.DataFrame(list(zip([p]*10, prompts))))
        print(prompts)

    print(paraphrased_templates)
    paraphrased_templates.to_csv('v2.11d.chatgpt_templates.csv')

def reparaphrase():
    df = pd.read_csv("gpt3paraphs.csv", index_col=0)
    df["score"] = df.apply(lambda row: 0 if pd.isnull(row["prompt"])  else 1,axis=1)
    gb = df.groupby("fid")
    df_scores = gb.sum("score")
    df_original = pd.read_csv("v2.11d.csv", index_col=0)
    gpt3 = GPT3()
    for i, fid in enumerate(df_scores.loc[df_scores["score"]<10].index):
        print("=======")
        print(fid)
        print(df[fid*10:fid*10+10])
        original_prompt = df_original.loc[fid]["prompt"]
        paraphs = gpt3.get_paraphrases(original_sentence=original_prompt)
        for j in range(10):
            df.loc[fid*10+j, "prompt"] = paraphs[j]
        print(df[fid*10:fid*10+10])
        df.to_csv("gpt3paraphs.csv")
    
    for i, fid in enumerate(df.loc[(df.index % 10 == 9) & (~df['prompt'].str.endswith('?'))]['fid']):
        print("=======")
        print(fid)
        print(df[fid*10:fid*10+10])
        original_prompt = df_original.loc[fid]["prompt"]
        paraphs = gpt3.get_paraphrases(original_sentence=original_prompt)
        for j in range(10):
            df.loc[fid*10+j, "prompt"] = paraphs[j]
        print(df[fid*10:fid*10+10])
        df.to_csv("gpt3paraphs.csv")

    df.to_csv("gpt3paraphs.csv")



    # print(df_scores.loc[df_scores["score"]<10].index)
    # print(df_scores)
    # print(df)
    # print(gb.loc[gb["score"]<10])

if __name__ == '__main__':
    # reparaphrase()
    gpt3 = GPT3()