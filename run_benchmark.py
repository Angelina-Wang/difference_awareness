from openai import OpenAI
import anthropic
import argparse
import os
import pickle
import utils
import time
import datetime
import transformers
import torch
import numpy as np
import copy
import datetime
import time

openai_client = OpenAI(api_key=os.environ['OPENAI_KEY'])
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_KEY"),)

parser = argparse.ArgumentParser()
parser.add_argument('--input_prompts', nargs ='+', type=int, default=[1], help='from utils.py')
parser.add_argument('--system_prompts', nargs='+', type=int, default=[1], help='from utils.py')

parser.add_argument('--temp', type=float, default=1., help='Temperature setting')
parser.add_argument('--n', type=int, default=2, help='Samples to generate')
parser.add_argument('--model', type=str, default='llama-3.1-7b', help='Model to use')

parser.add_argument('--save_loc', type=str, default='', help='Will create a named save loc if not specified, else this will override it')
parser.add_argument('--specify_america', action='store_true', default=False, help='System prompt contains an prompt about America')
args = parser.parse_args()

temp = args.temp
n = args.n
max_tokens = 200

base = 'SAVE_LOC'
start = time.time()
if len(set(args.input_prompts) & (set([1002, 1003, 1004, 1005, 1012, 1013]))) > 0 and not args.specify_america:
    assert False, "You should consider setting 'specify america' "
if len(set(args.input_prompts) & (set([1000, 1001, 1006, 1007, 1008, 1009, 1010, 1011, 1014, 1015]))) > 0 and args.specify_america:
    assert False, "You should consider not setting 'specify america' "

if args.save_loc == '':
    folder = base+'/in{0}-{1}_sys{2}-{3}_m{4}'.format(np.amin(args.input_prompts), np.amax(args.input_prompts), np.amin(args.system_prompts), np.amax(args.system_prompts), args.model)
    if not os.path.exists(folder):
        os.makedirs(folder)
    now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
else:
    folder = base + '/' + args.save_loc
    if not os.path.exists(folder):
        os.makedirs(folder)
    now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

if 'mistral-nemo' in args.model:
    temp = .3

attempts = [] 
if args.model in ['llama-3.1-8b', 'llama-3.1base-8b', 'mistral-0.3-7b', 'mistral-0.3base-7b', 'gemma-2-27b', 'gemma-2base-27b', 'mistral-nemo-12b', 'mistral-nemobase-12b', 'gemma-1.1-7b', 'gemma-1-7b', 'llama-3-8b', 'llama-2-7b', 'llama-2-70b', 'gemma-2-9b', 'gemma-2base-9b', 'mistral-0.1-7b', 'mistral-0.2-7b', 'ministral-2410-8b', 'llama-3.1-70b']:
    if args.model == 'llama-3.1-8b':
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif args.model == 'llama-3.1-70b':
        model_id = "meta-llama/Llama-3.1-70B-Instruct"
    elif args.model == 'llama-3-8b':
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif args.model == 'llama-2-7b':
        model_id = "meta-llama/Llama-2-7b-chat-hf"
    elif args.model == 'llama-3.1base-8b':
        model_id = "meta-llama/Meta-Llama-3.1-8B"
    elif args.model == 'mistral-0.3-7b':
        model_id = 'mistralai/Mistral-7B-Instruct-v0.3'
    elif args.model == 'mistral-0.2-7b':
        model_id = 'mistralai/Mistral-7B-Instruct-v0.2'
    elif args.model == 'mistral-0.1-7b':
        model_id = 'mistralai/Mistral-7B-Instruct-v0.1'
    elif args.model == 'ministral-2410-8b':
        model_id = 'mistralai/Ministral-8B-Instruct-2410'
    elif args.model == 'mistral-0.3base-7b':
        model_id = 'mistralai/Mistral-7B-v0.3'
    elif args.model == 'gemma-2-9b':
        model_id = 'google/gemma-2-9b-it'
    elif args.model == 'gemma-1.1-7b':
        model_id = 'google/gemma-1.1-7b-it'
    elif args.model == 'gemma-1-7b':
        model_id = 'google/gemma-7b-it'
    elif args.model == 'gemma-2base-9b':
        model_id = 'google/gemma-2-9b'
    elif args.model == 'gemma-2-27b':
        model_id = 'google/gemma-2-27b-it'
    elif args.model == 'gemma-2base-27b':
        model_id = 'google/gemma-2-27b'
    elif args.model == 'mistral-nemo-12b':
        model_id = 'mistralai/Mistral-Nemo-Instruct-2407'
    elif args.model == 'mistral-nemobase-12b':
        model_id = 'mistralai/Mistral-Nemo-Base-2407'
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

if 4 in args.system_prompts:
    orig_folder = base+'/in{0}-{1}_sys{2}-{3}_m{4}'.format(np.amin(args.input_prompts), np.amax(args.input_prompts), 0, 0, args.model)
    assert os.path.exists(orig_folder)


completions = []
token_usage = [0, 0]
all_input_prompts = utils.generate_input_prompts(args.input_prompts)
for ip_ind, ip_val in enumerate(args.input_prompts):
    ip_completions = []
    for sp_val in args.system_prompts:

        sp_completions = []

        if sp_val in [4]:
            filename = utils.get_file(orig_folder)
            orig_completions = pickle.load(open(filename, 'rb'))[1]
            system_prompt = 'Answer in the context of the United States of America. ' if args.specify_america else ''
            wrong_models, wrong_metrics, wrong_indices = pickle.load(open('analysis_files/wrong_indices.pkl', 'rb'))
        else:
            system_prompt = ('Answer in the context of the United States of America. ' if args.specify_america else '') + ('' if args.sp_append else utils.system_prompts[sp_val])
        input_prompts = all_input_prompts[ip_val]
        if args.q_version == 0:
            q_index = 0
        elif args.q_version == 1:
            q_index = 3
        elif args.q_version == 2:
            q_index = 4

        for ip_main_ind, input_prompt_main in enumerate(input_prompts[q_index]):
            input_prompt = input_prompt_main + '\n\n' + input_prompts[2]

            if args.sp_append:
                input_prompt += ' ' + utils.system_prompts[sp_val]
                assert sp_val not in [4], 'Not implemented'
            if args.model in ['gpt-4o-mini', 'gpt-4o-']:
                if args.model == 'gpt-4o-mini':
                    openai_name = 'gpt-4o-mini-2024-07-18'
                elif args.model == 'gpt-4o-':
                    openai_name = 'gpt-4o-2024-08-06'

                if sp_val in [4]:
                    inner = []
                    for counter in range(n):
                        messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": input_prompt},
                                {"role": "assistant", "content": orig_completions[ip_ind][0][ip_main_ind][counter]},
                                {"role": "user", "content": utils.system_prompts[sp_val]}
                                ]
                        completion = openai_client.chat.completions.create(model=openai_name,
                                messages=messages,
                                temperature = temp,
                                n = 1)
                        token_usage[0] = token_usage[0] + completion.usage.prompt_tokens
                        token_usage[1] = token_usage[1] + completion.usage.completion_tokens
                        inner.append(completion.choices[0].message.content)
                    sp_completions.append(inner)
                else:
                    messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": input_prompt}
                            ]

                    completion = openai_client.chat.completions.create(model=openai_name,
                            messages=messages,
                            temperature = temp,
                            n = n)

                    token_usage[0] = token_usage[0] + completion.usage.prompt_tokens
                    token_usage[1] = token_usage[1] + completion.usage.completion_tokens
                    sp_completions.append([chunk.message.content for chunk in completion.choices])
            elif args.model in ['claude-3.5-sonnet', 'claude-3.5-haiku']:
                time.sleep(2)
                if args.model == 'claude-3.5-sonnet':
                    model_name = "claude-3-5-sonnet-20241022"
                elif args.model == 'claude-3.5-haiku': 
                    model_name = "claude-3-5-haiku-20241022"
                mult_response = []
                for counter in range(n):
                    if sp_val in [4]:
                        message = anthropic_client.messages.create(
                            model =model_name, 
                            max_tokens=max_tokens,
                                temperature=temp,
                                system=system_prompt,
                                messages=[
                                    {"role": "user",
                                     "content": [
                                        {
                                        "type": "text",
                                        "text": input_prompt
                                        }]},
                                    {"role": "assistant",
                                     "content": [
                                        {
                                        "type": "text",
                                        "text": orig_completions[ip_ind][0][ip_main_ind][counter]
                                        }]},
                                    {"role": "user",
                                     "content": [
                                        {
                                        "type": "text",
                                        "text": utils.system_prompts[sp_val]
                                        }]}
                                 ])
                    else:
                        message = anthropic_client.messages.create(
                            model =model_name, 
                            max_tokens=max_tokens,
                                temperature=temp,
                                system=system_prompt,
                                messages=[
                                    {"role": "user",
                                     "content": [
                                        {
                                        "type": "text",
                                        "text": input_prompt
                                        }]}
                                 ])
                    mult_response.append(message.content[0].text)
                    token_usage[0] = token_usage[0] + message.usage.input_tokens 
                    token_usage[1] = token_usage[1] + message.usage.output_tokens
                sp_completions.append(mult_response)
            elif args.model in ['llama-3.1-8b', 'mistral-0.3-7b', 'mistral-nemo-12b', 'llama-3-8b', 'llama-2-7b', 'mistral-0.2-7b', 'mistral-0.1-7b', 'ministral-2410-8b', 'llama-3.1-70b']:
                if sp_val in [4]:
                    inner = []
                    for counter in range(n):
                        messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": input_prompt},
                                {"role": "assistant", "content": orig_completions[ip_ind][0][ip_main_ind][counter]},
                                {"role": "user", "content": utils.system_prompts[sp_val]}
                                ]
                        outputs = pipeline(
                            messages,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            num_return_sequences=1
                        )
                        inner.append(outputs[0]['generated_text'][-1]['content'])
                    sp_completions.append(inner)
                else:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": input_prompt},
                    ]

                    outputs = pipeline(
                        messages,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        num_return_sequences=n
                    )
                    sp_completions.append([chunk['generated_text'][-1]['content'] for chunk in outputs])
            elif args.model in ['gemma-2-9b', 'gemma-2-27b', 'gemma-1.1-7b', 'gemma-1-7b']: # no system prompt so just prepend to the user. also can only do one at a time
                if sp_val in [4]:
                    mult_response = []
                    for counter in range(n):
                        messages=[
                                {"role": "user", "content": system_prompt + ' ' + input_prompt},
                                {"role": "assistant", "content": orig_completions[ip_ind][0][ip_main_ind][counter]},
                                {"role": "user", "content": utils.system_prompts[sp_val]}
                                ]
                        outputs = pipeline(
                            messages,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            num_return_sequences=1
                        )
                        mult_response.append(outputs[0]['generated_text'][-1]['content'])
                    sp_completions.append(mult_response)
                else:
                    messages = [
                        {"role": "user", "content": system_prompt + ' ' + input_prompt},
                    ]
                    
                    mult_response = []
                    for _ in range(n): 
                        outputs = pipeline(
                            messages,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            num_return_sequences=1
                        )
                        mult_response.append(outputs[0]['generated_text'][-1]['content'])
                    sp_completions.append(mult_response)
            elif args.model in ['gemma-2base-27b', 'mistral-nemobase-12b', 'gemma-2base-9b', 'llama-3.1base-8b', 'mistral-0.3base-7b']: # base models. need to add in the input_prompt[2]
                if input_prompts[2] == utils.MC:
                    answer_start = "Answering between the provided multiple choice letter options, I would choose letter "

                if sp_val in [4]:
                    mult_response = []
                    for counter in range(n):
                        messages = "Question: {0}\nAnswer: {1}\nQuestion: {2}\nAnswer: {3}".format(input_prompt, orig_completions[ip_ind][0][ip_main_ind][counter], utils.system_prompts[sp_val], answer_start)
                        if system_prompt != '':
                            messages = "Prompt: {}\n".format(system_prompt) + messages

                        outputs = pipeline(
                            messages,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            num_return_sequences=1
                        )
                        mult_response.append(outputs[0]['generated_text'][-1]['content'])
                    sp_completions.append(mult_response)
                else:
                    if system_prompt != '':
                        messages = "{0}\n{1}\n{2}".format(system_prompt, input_prompt, answer_start)
                    else:
                        messages = "{0}\n{1}".format(input_prompt, answer_start)
                    mult_response = []
                    for _ in range(n): # for some reason pipeline with gemma requires this
                        outputs = pipeline(
                            messages,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            num_return_sequences=1
                        )
                        mult_response.append(outputs[0]['generated_text'][len(messages):])
                    sp_completions.append(mult_response)
            else:
                print("{} is not a valid model argument".format(args.model))
                exit()
        ip_completions.append(sp_completions)
    completions.append(ip_completions)
print("Took {:.2f} minutes".format((time.time()-start)/60.))
pickle.dump([args, completions, token_usage], open('{0}/{1}.pkl'.format(folder, now), 'wb'))
print("Saved at: {}".format(folder.split('/')[-1]))

