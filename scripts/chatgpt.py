import openai
import os
import sys
import time
import re
import pandas as pd
class ChatGPT():
    def __init__(self):
        self.model = "text-davinci-003" # GPT-3 engine here
        self.max_tokens=256
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

def str2list(str):
    pattern = re.compile(r'^[0-9]+. (.+)')
    prompts = str.strip().split('\n')
    prompts = [pattern.match(prompt).group(1) for prompt in prompts]
    return prompts

### Let's look at a simple QA for GPT3.

def main():
    facts = pd.read_csv('v2.11d.csv', index_col=0)
    facts = facts.drop_duplicates('p', keep='first')
    facts['template'] = facts.apply(lambda row: row['prompt'].replace(row['ss'], '[MASK]'), axis=1)
    templates = facts['template'].to_list()
    pids = facts['p'].to_list()

    chatgpt = ChatGPT()
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

if __name__ == '__main__':
    main()