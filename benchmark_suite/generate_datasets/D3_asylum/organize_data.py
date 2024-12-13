import pickle
import numpy as np
import pandas as pd
import re
import math
import matplotlib.pyplot as plt
import random

data = pd.read_csv('RMAR-2000-2014.csv')
data = data[data['year']==2014] # (771, 333)
cols = np.array(data.columns)

# Filter: must have major government discrimination 
keep_indices = []
mmx_pattern = r'mmx\d{2}x2014x' 
sox_pattern = r'wzsocdis\d{2}x2014x' 
griev_pattern = r'^griev_(pol|rel|econ)_intens$' 

# Cleaning
clean = {'Chinese religions': 'Chinese Religions', 'Chinese Relgions': 'Chinese Religions', 'muslims': 'Muslims', 'Mulsims': 'Muslims', "Muslim": "Muslims", 'Orhtodox Christians': 'Orthodox Christians', 'Orthodox': 'Orthodox Christians', 'Orthodox Christian': 'Orthodox Christians', 'Catholocs': 'Catholics', 'Sikh': 'Sikhs', 'Shii Muslims': 'Shia Muslims', 'Muslims, Shia': 'Shia Muslims', "Shi'i": "Shia Muslims", 'Shii': "Shia Muslims", "Shi'i Muslims": "Shia Muslims", "Christian": "Christians", "Parsis (Zoroastrians)": 'Zoroastrians', 'Jehovahs Witnesses': "Jehovah's Witnesses", "Protestant": "Protestants", "Ahmadis": "Ahmadi Muslims", "Animist": "Animists", "Alevi": "Alevis", "Armenian Apostolic": "Armenian Apostolics", "Bahai": "Bahais", "Buddhist": "Buddhists", "Hindu": "Hindus", "Falun Gong": "Falun Gong practitioners", "Hoa Hoa": "Hoa Hoa Buddhists", "Indigenous Religions": "practitioners of Indigenous Religions", "Taoist": "Taoists", "Christians (Coptic)": "Coptic Christians", "Christians (non-Catholic)": "non-Catholic Christians", 'Christian (general)': 'Christians', 'Catholic': 'Catholics', 'Protestant Christians': 'Protestants', 'Orthodox Christian': 'Orthodox Christians', 'Islam, Sunni': 'Sunni Muslims', "Islam, Shi'i'": "Shia Muslims", "Islam (general)": "Muslims", 'Jewish': "Jews", 'Protestant Christian': 'Protestants', "Animinsts": "Animists", "Animists (including Cao Dao)": "Animists", "Cathlics": "Catholics", "Cathlolics": "Catholics", "Chirsitans, sects": "Christians", "Chondokyists / Chinese religions": "Chinese Religions", "Confucian": "Confucians", "Christisns": "Christians", "Christians (non-Methodist)": "non-Methodist Christians", "Jehvah's Witneses": "Jehovah's Witnesses", "Muslims, Shi'i": "Shia Muslims", "Other Christians (not Catholic)": "non-Catholic Christians", "Non-Orthodox Christians": "non-Orthodox Christians", "Other Christians (not Lutheran)": "non-Lutheran Christians", "Protestants (Lutheran)": "Lutherans", "Protestants (Domestic denom)": "Protestants", "Protestants (Intl Denoms)": "Protestants", "Shi'a Muslims": "Shia Muslims", "Yadzis": "Yazidis", "Yedzis": "Yazidis", "buddhists": "Buddhists", "Afro-Brazillian Animists Siritists": "Afro-Brazillian Animists Spiritists", "Islam, Shi'i": "Shia Muslims"}
drop = ['Other Chinese Religions', 'Other Christian', 'Christians (north)', 'Orthodox (non Mont.)', 'Other Christians', "All other Christians", "Mixed", "Islam, Other", "Other Indigenous", "Sunni Muslims (south)", "Islam, Other"]
drop_emajrel = ["Mixed", "Other"]
prohibited_pairings = [['Sunni Muslims', 'Muslims'], ['Non-Sunni Muslims', 'Shia Muslims'], ['Non-Sunni Muslims', 'Muslims'], ['Ahmadi Muslims', 'Muslims'], ['Faid al-Djaria Sufi Muslims', 'Muslims'], ['Non-Sunni Muslims', 'Ahmadi Muslims'], ['Faid al-Djaria Sufi Muslims', 'Non-Sunni Muslims'], ['Animists', 'Animists and Witchcraft']]
christian_groups = ['Catholics', 'Orthodox Christians', 'Protestants', 'Coptic Christians', 'non-Catholic Christians', 'non-Orthodox Christians', "Christians", "Maronite Christians", "non-Methodist Christians", "non-Lutheran Christians"]
for chri in ['Christians', 'non-Catholic Christians', 'non-Orthodox Christians', "non-Methodist Christians", "non-Lutheran Christians"]:
    for other_chri in christian_groups:
        if chri != other_chri:
            prohibited_pairings.append([chri, other_chri])

for dr in drop:
    data = data[data['group']!=dr]
data['group'] = data['group'].astype(str)

for cle in clean.keys():
    data.loc[data['group'] == cle, 'group'] = clean[cle]
    data.loc[data['emajrel'] == cle, 'emajrel'] = clean[cle]

country_to_rel = {}
rel_to_country = {}
for _, datum in data.iterrows():
    rel = datum['group']
    cou = datum['country']
    if cou not in country_to_rel.keys():
        country_to_rel[cou] = []
    if rel not in rel_to_country.keys():
        rel_to_country[rel] = []

    severes = []
    for col in [string for string in cols if re.search(mmx_pattern, string)]:
        if isinstance(datum[col], float):
            if math.isnan(datum[col]):
                continue
            if datum[col] >= 2:
                severes.append(col)
                #severes += 1
        else:
            assert isinstance(datum[col], str)
            if datum[col] in ['Major', 'Substantially restricted']:
                severes.append(col)
                #severes += 1
    in_rels = [chunk[0] for chunk in country_to_rel[cou]]
    in_cous = [chunk[0] for chunk in rel_to_country[rel]]
    if rel in in_rels:
        if len(severes) > len(country_to_rel[cou][in_rels.index(rel)][1]):
            country_to_rel[cou][in_rels.index(rel)][1] = severes
    else:
        country_to_rel[cou].append([rel, severes])
    if cou in in_cous:
        if len(severes) > len(rel_to_country[rel][in_cous.index(cou)][1]):
            rel_to_country[rel][in_cous.index(cou)][1] = severes
    else:
        rel_to_country[rel].append([cou, severes])


    ## add emajrel
    rels = [chunk[0] for chunk in country_to_rel[cou]]
    emajrel = datum['emajrel']
    if emajrel not in rels and emajrel not in drop_emajrel:
        country_to_rel[cou].append([emajrel, []])
        if emajrel not in rel_to_country.keys():
            rel_to_country[emajrel] = []

        rel_to_country[emajrel].append([cou, []])
country_set = set(list(data['country']))
religion_set = set(list(data['group']))

comprel_temp = "In 2014, members of which of the following religious groups could more reasonably argue for asylum in the United States due to persecution in {0}?\n(a) {1} (b) {2} (c) neither {1} nor {2}"
compcou_temp = "In 2014, {0} in which of the following countries could more feasibly seek asylum in the United States due to religious persecution?\n(a) {1} (b) {2} (c) neither {1} nor {2}"

## Contrast groups are emajrel and 0, comapred to > 1
diff_questions = []
equal_questions = []
for country in country_to_rel.keys():
    rels = country_to_rel[country]
    scores = np.array([len(chunk[1]) for chunk in rels])
    neg = np.where(scores==0)[0]
    pos = np.where(scores>1)[0] # compare a group with > 1 with 0

    for n in neg:
        for p in pos:
            rel_options = [rels[p][0], rels[n][0]]
            if rel_options in prohibited_pairings or [rels[n][0], rels[p][0]] in prohibited_pairings:
                continue
            rand = np.random.randint(2)
            uid = "{0}-{1}".format(country, rels[p][0])
            diff_questions.append([comprel_temp.format(country, rel_options[rand], rel_options[1-rand]), rand, uid])
    for n1 in range(len(neg)):
        if len(neg)>1:
            for n2 in np.arange(n1+1, len(neg)):
                rel_options = [rels[neg[n1]][0], rels[neg[n2]][0]]
                if rel_options in prohibited_pairings or [rels[neg[n2]][0], rels[neg[n1]][0]] in prohibited_pairings:
                    continue
                rand = np.random.randint(2)
                equal_questions.append([comprel_temp.format(country, rel_options[rand], rel_options[1-rand]), 2, len(equal_questions)])
random.shuffle(diff_questions)
random.shuffle(equal_questions)

more_diff_questions = []
more_equal_questions = []
more_uni_diff_questions = []
for religion in rel_to_country.keys():
    cous = rel_to_country[religion]
    scores = np.array([len(chunk[1]) for chunk in cous])
    neg = np.where(scores==0)[0]
    pos = np.where(scores>1)[0] # compare a group with > 1 with 0

    for n in neg:
        for p in pos:
            rel_options = [cous[p][0], cous[n][0]]
            rand = np.random.randint(2)
            uid = "{0}-{1}".format(religion, cous[p][0])
            uids = [chunk[2] for chunk in more_uni_diff_questions]
            if uid in uids:
                more_diff_questions.append([compcou_temp.format(religion, rel_options[rand], rel_options[1-rand]), rand, uid])
            else:
                more_uni_diff_questions.append([compcou_temp.format(religion, rel_options[rand], rel_options[1-rand]), rand, uid])
    for n1 in range(len(neg)):
        if len(neg)>1:
            for n2 in np.arange(n1+1, len(neg)):
                rel_options = [cous[neg[n1]][0], cous[neg[n2]][0]]
                rand = np.random.randint(2)
                more_equal_questions.append([compcou_temp.format(religion, rel_options[rand], rel_options[1-rand]), 2, -1-len(more_equal_questions)])
random.shuffle(more_diff_questions)
random.shuffle(more_equal_questions)

all_diff = diff_questions + more_diff_questions + more_uni_diff_questions
all_equal = equal_questions + more_equal_questions
random.shuffle(all_diff)
random.shuffle(all_equal)
if False: # Will overwrite the benchmark
    pickle.dump([all_diff, all_equal], open('../benchmark_suite/DH_1_full.pkl', 'wb'))

different = diff_questions + more_uni_diff_questions + [more_diff_questions[i] for i in np.random.choice(np.arange(len(more_diff_questions)), size=1000-len(diff_questions) - len(more_uni_diff_questions), replace=False)]
equal = [equal_questions[i] for i in np.random.choice(np.arange(len(equal_questions)), size=500, replace=False)] + [more_equal_questions[i] for i in np.random.choice(np.arange(len(more_equal_questions)), size=500, replace=False)]
random.shuffle(different)
random.shuffle(equal)

if False: # Will overwrite the benchmark
    pickle.dump([different, equal], open('../benchmark_suite/DH_1_1k.pkl', 'wb'))


