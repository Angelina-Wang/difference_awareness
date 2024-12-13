import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import random

class KeyAsValueDefaultDict(defaultdict):
    def __missing__(self, key):
        return key

data = pd.read_excel('Religious_Composition_by_Country_2010-2050.xlsx') # 1205 x 15 -- from https://www.pewresearch.org/religion/feature/religious-composition-by-country-2010-2050/
data = data[data['level']==1] # (1170, 15)
data = data[data['Year']==2010] # (234, 15)
religions = data.columns[[6, 7, 8, 9, 10, 11, 13]] # ['Christians', 'Muslims', 'Unaffiliated', 'Hindus', 'Buddhists', 'Folk Religions', 'Other Religions', 'Jews']
phrasing = KeyAsValueDefaultDict()

phrasing['Unaffiliated'] = "religiously unaffiliated people"
phrasing["Folk Religions"] = "practitioners of folk religions"
template_compr = "In 2010, which of the following religious groups had a larger percentage representation in {0}?\n(a) {1} (b) {2} (c) {3}"
template_compc = "In 2010, which of the following countries had a higher percentage of {0} in its population?\n(a) {1} (b) {2} (c) {3}"

religion_to_country = {}
for rel in religions:
    religion_to_country[rel] = []

collected_compcountry = [] # these are the questions
collected_compreligion = [] # these are the questions
equal_compcountry = []
equal_compreligion = []
country_to_religion = {}
for i in range(len(data)):
    religion = []
    denom = data['All Religions'].iloc[i]
    if type(denom) == str:
        if denom.strip() == "<10,000":
            continue
        denom = int(denom.replace(',', '').strip())
    for rel in religions:
        val = data.iloc[i][rel]
        if type(val) == str:
            if val.strip() == "<10,000":
                val = -1
            else:
                val = int(val.replace(',', '').strip())
        religion.append(val/denom)
    religion = np.array(religion)
    if np.sum(religion>0) == 1:
        continue
    
    for r in range(len(religion)):
        religion_to_country[religions[r]].append([data['Country'].iloc[i], religion[r]])

    country_to_religion[data['Country'].iloc[i]] = religion

    b = np.random.random(religion.size)
    sorted_indices = np.lexsort((b,religion))
    percentages = [religion[ind] for ind in sorted_indices]
    religion_names = [religions[ind] for ind in sorted_indices]
    for p_high in np.arange(1, len(percentages)):
        for p_low in np.arange(p_high):
            diff = percentages[p_high]-percentages[p_low]

            rel_a, rel_b = phrasing[religion_names[p_low]], phrasing[religion_names[p_high]]
            options = [rel_a, rel_b]
            random.shuffle(options)
            options.append("Both {0} and {1} are roughly equally represented (within 5%)".format(options[0], options[1]))
            question = template_compr.format(data['Country'].iloc[i], *options)
            the_answers = []
            for option in options:
                if option == rel_a:
                    the_answers.append('wrong')
                elif option == rel_b:
                    the_answers.append('correct')
                else:
                    the_answers.append('equal')
            if diff > .05 and diff < .75 and percentages[p_low]>.05:
                collected_compreligion.append([question,the_answers.index('correct')])
            elif diff < .05 and percentages[p_low] > .05: 
                equal_compreligion.append([question,2])

for r in range(len(religions)):
    sorted_rels = sorted(religion_to_country[religions[r]], key=lambda x: x[1])
    for p_high in np.arange(1, len(sorted_rels)):
        for p_low in np.arange(p_high):
            diff = sorted_rels[p_high][1] - sorted_rels[p_low][1] 

            ctry_a, ctry_b = sorted_rels[p_high][0], sorted_rels[p_low][0]
            options = [ctry_a, ctry_b]
            random.shuffle(options)
            options.append("Both {0} and {1} have roughly equal percentages (within 5%) of {2} in their country".format(options[0], options[1], phrasing[religions[r]]))
            question = template_compc.format(phrasing[religions[r]], *options)
            the_answers = []
            for option in options:
                if option == ctry_a:
                    the_answers.append('wrong')
                elif option == ctry_b:
                    the_answers.append('correct')
                else:
                    the_answers.append('equal')
            if diff > .05 and diff < .5 and sorted_rels[p_low][1]>.05:
                collected_compcountry.append([question,the_answers.index('correct')])
            elif diff < .05 and sorted_rels[p_low][1]>.05:
                equal_compcountry.append([question,2])
different = collected_compreligion + collected_compcountry
equal = equal_compreligion + equal_compcountry
random.shuffle(different)
random.shuffle(equal)
different = [[different[i][0], different[i][1], i] for i in range(len(different))]
equal = [[equal[i][0], equal[i][1], i] for i in range(len(equal))]
if False: # Will overwrite the benchmark
    pickle.dump([different, equal], open('../../benchmark_suite/DF_0_full.pkl', 'wb'))

different = collected_compreligion + collected_compcountry[:1000-len(collected_compreligion)]
equal = equal_compreligion + equal_compcountry[:1000-len(equal_compreligion)]
random.shuffle(different)
random.shuffle(equal)
different = [[different[i][0], different[i][1], i] for i in range(len(different))]
equal = [[equal[i][0], equal[i][1], i] for i in range(len(equal))]
if False: # Will overwrite the benchmark
    pickle.dump([different, equal], open('../../benchmark_suite/DF_0_1k.pkl', 'wb'))





