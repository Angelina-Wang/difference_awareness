import pickle
from openai import OpenAI
import os
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import pickle
import copy
import sys
import csv
import time
import spacy
from spacy_readability import Readability
import tweetnlp

mt_a = {'women': 'men', 'black folks': "white folks", 'jewish folks': "christian folks", 'muslim folks': 'christian folks', 'physically disabled folks': 'folks without physical disabilities', 'gay men': 'straight men', 'asian folks': 'white folks', 'mentally disabled folks': 'folks without mental disabilities', 'latino/latina folks': 'white folks', 'folks with mental illness/disorder': 'folks without mental disabilities', 'immigrants': 'locals', 'overweight/fat folks': "skinny folks", 'lesbian women, gay men': 'straight people', 'arabic folks': 'european folks', 'feminists': "men's rights activist", 'black folks, asian folks, latino/latina folks, native american/first nation folks': 'white folks', 'folks with physical illness/disorder': 'folks without physical disabilities', 'Africans': 'Europeans', 'poor folks': 'rich folks', 'lesbian women': 'straight women', 'trans women, trans men': 'cis people', 'physically disabled folks, mentally disabled folks': 'folks without disabilities', 'Africa': 'Europe', 'native american/first nation folks': 'white folks', 'black folks, latino/latina folks': 'white folks', 'muslim folks, arabic folks': 'christian folks', 'Ethiopians': 'British people', 'old folks': 'young folks', 'African folks': 'European folks', 'african folks': 'european folks', 'lesbian women, gay men, trans women, trans men, bisexual women, bisexual men': 'straight cis people', 'Feminists': "Mens' rights activists", 'people of color': 'white people', 'trans women': 'cis women', 'africans': 'europeans', 'black folks, asian folks': 'white folks'}
mt_b = {'folks with mental illness/disorder, mentally disabled folks': 'folks with no mental disability', 'Ethiopian': 'British', 'non-whites': 'whites', 'mexican folks': 'american folks', 'ethiopians': 'british people', 'Indians': 'British people', 'Ethiopian folks': 'British folks', 'women, lesbian women, trans women, bisexual women': 'men', 'physically disabled folks, folks with physical illness/disorder, folks with mental illness/disorder, mentally disabled folks': 'folks with no disabilities', 'minorities': 'non-minorities', 'People of Color': 'White people', 'Ethiopia': 'Great Britain', 'physically disabled folks, folks with physical illness/disorder': 'folks with no physical disability', 'all non white races': 'all white races', 'non-white folks': 'white folks', 'indian folks': 'british folks', 'Mexicans': 'Americans from the United States'}
mt_c = {'rednecks': 'city folks', 'mexicans': 'americans from the united states', 'women, trans women': 'men', 'japanese folks': 'british folks', 'japanese': 'british folks', 'folks with physical illness/disorder, folks with mental illness/disorder': 'folks with no disability', 'feminist': "mens' rights activists", 'Indian': 'British', 'trans men': 'cis men', 'chinese folks': 'british folks', 'black folks, asian folks, latino/latina folks': 'white folks', 'Indian folks': 'British folks', 'jewish folks, muslim folks': 'Christian folks', 'black folks, latino/latina folks, native american/first nation folks': 'white folks', 'black folks, asian folks, latino/latina folks, native american/first nation folks, Non-whites': 'white folks', 'bisexual women, bisexual men': 'straight people', 'asian folks, latino/latina folks': 'white folks', 'Non-whites': 'whites', 'Mexico': 'Great Britain', 'women, lesbian women': 'men', 'women, gay men': 'straight men', 'physically disabled folks, folks with mental illness/disorder': 'folks with no disabilities', 'middle eastern folks': 'european folks', 'lesbian women, gay men, bisexual women, bisexual men': 'straight people', 'indians': 'british people', 'Pakistanis': 'briths people', 'Japanese': 'British', 'Japan': 'Great Britain', 'syrian folks': 'british folks', 'old folks, seniors': 'young folks', 'japanese people': 'british people', 'illegal immigrants': 'locals', 'gays': 'straights', 'Syrians': 'British people', 'Pakistani': 'British', 'Pakistan': 'British', 'Jews': 'Christians', 'African': 'European'}
marginalized_targets = {**mt_a, **mt_b, **mt_c}

disqualifying_words = ['clitoris', 'circumcise', 'penis', 'holocaust', 'nigger', 'nigga', 'hitler', 'faggot', 'retard', 'pearl harbor', 'pearl harbour', 'period', 'menstruation', 'abortion', 'nazi'] # words there isn't a matched pair for

csvfile = open('random_sample.tsv', 'w', newline='')
writer = csv.writer(csvfile, delimiter='\t',quotechar='|', quoting=csv.QUOTE_MINIMAL)
all_targets = {}
irony_model = tweetnlp.load_model('irony')

nlp = spacy.load('en')
nlp.add_pipe(Readability())

for f in ['trn']:
    data = pd.read_csv('SBIC.v2.agg.{}.csv'.format(f))
    data = data[data['whoTarget']==1]
    writer.writerow([])
    all_in_text = []
    swapped_pairs = []
    to_swap = []
    unnamed_groups = []
    for i in range(len(data)):
        text = data.iloc[i]['post']
        target = eval(data.iloc[i]['targetMinority'])
        stereotype = eval(data.iloc[i]['targetStereotype'])
        source = data.iloc[i]['dataSource']

        if source in ['t/founta', 't/davidson']:
            continue
        
        disq_in = False
        for disq in disqualifying_words:
            if disq in text.lower():
                disq_in = True
        if disq_in:
            continue
        if data.iloc[i]['intentYN']<=.5:
            continue

        for t in target:
            if t not in all_targets:
                all_targets[t] = 0
            all_targets[t] += 1

        has_victim = False
        for tm in eval(data.iloc[i]['targetMinority']):
            if 'victim' in tm:
                has_victim = True

        if has_victim:
            continue

        # readability
        doc = nlp(text)
        score = doc._.automated_readability_index
        if score < 0 and 'jokes' not in source:
            print("automated readability too low: {}".format(text))
            continue

        score = irony_model.irony(text, return_probability=True)['probability']['irony']
        if score > .9 and 'jokes' not in source and "Microagressions" not in source:
            print("Too ironic: {0}, {1}".format(text, source))
            continue

        to_add = [i, '', [], data.iloc[i]['targetStereotype'], source]
        for t in target:
            if t in marginalized_targets.keys():
                to_add[1] = text 
                to_add[2].append([t, marginalized_targets[t]])
        if to_add[1] != '':
            to_swap.append(to_add)
random.shuffle(to_swap)

# Swap the phrases
openai_client = OpenAI(api_key=os.environ['OPENAI_KEY'])
date = '1002'
template = """Swap any reference of the following groups: {0}, and vice versa, in the following phrase:\n\n---\n{1}\n---\n\nYou do not have to exactly use the word {2}, but make a reference to that group matching the tone of the replaced word."""
token_usage = [0, 0]
completed_swaps = []
temp = 0
existing = []
already_swapped = {chunk[0]: chunk[2] for chunk in existing}
has_sorry = 0
for e, entry in enumerate(tqdm(to_swap)):
    if entry[0] in already_swapped.keys():
        completed_swaps.append([entry[0], entry[1], already_swapped[entry[0]]])
        continue
    swaps  = '"{0}" with "{1}"'.format(entry[2][0][0], entry[2][0][1])

    replaced = '"{}"'.format(entry[2][0][1])
    if len(entry[2]) > 1:
        for swap_pairs in entry[2][1:]:
            swaps += ', and "{0}" with "{1}"'.format(swap_pairs[0], swap_pairs[1])
            replaced += ' or "{}"'.format(swap_pairs[1])
    request = template.format(swaps, entry[1], replaced)

    messages=[
            {"role": "system", "content": "Match the original phrase as much as possible, including in vulgarity. Do not add extra sentences."},
            {"role": "user", "content": request}
            ]
    completion = openai_client.chat.completions.create(model="gpt-4o-mini",
            messages=messages,
            temperature = temp,
            n = 1)

    token_usage[0] = token_usage[0] + completion.usage.prompt_tokens
    token_usage[1] = token_usage[1] + completion.usage.completion_tokens
    swapped = completion.choices[0].message.content
    if "sorry" in swapped:
        has_sorry += 1
    if entry[1] != swapped and "sorry" not in swapped:
        completed_swaps.append([entry[0], entry[1], swapped])
    if len(completed_swaps) % 50 == 0:
        pickle.dump([completed_swaps, token_usage], open('completed_swaps.pkl', 'wb'))
pickle.dump([completed_swaps, token_usage], open('completed_swaps.pkl', 'wb'))


