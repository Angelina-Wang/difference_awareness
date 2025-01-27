import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import pickle
import sys
import numpy as np
import analysis_utils
import time
sys.path.append('..')
import utils

file_name = 'all'
start = time.time()
models = utils.ALL_MODELS
metrics = ['fair-bbqamb', 'fair-bbqdis', 'fair-diseval', 'our-religion', 'our-occup-descript', 'our-legal', 'our-asylum', 'our-sbf', 'our-bbq', 'our-occup-affirmact', 'our-cultapp'] 
base = utils.BASE_FOLDER + '/' +utils.SAVE_FOLDER

data = {}
data['group'] = models
intervals = {}
steerability = {}
steerability_diffprompts = {}
extra_steervals = {}

data_diffprompts = {}
data_diffprompts['group'] = models

need_to_fix = {}
need_to_ref = {}
need_to_denom = {}

all_diffawares = {}
all_ctxtawares = {}
all_dists = {}

embedder = utils.Embedder()
sys_prompt_dict = {}
sys_prompt_dict['1-4'] = [1, 2, 3, 4]
sys_prompt_dict['5-7'] = [5, 6, 7]
sys_prompt_dict['0-0'] = [0]
sys_prompts = np.sort(list(sys_prompt_dict.keys()))

prompt_to_ind = {}
for val in np.sort(list(set(np.concatenate(list(sys_prompt_dict.values()))))):
    prompt_to_ind[val] = len(prompt_to_ind)

for mod_ind, model in enumerate(models):
    print("--{}--".format(model))
    need_to_fix[model] = np.zeros((len(metrics), len(prompt_to_ind)))
    need_to_ref[model] = np.zeros((len(metrics), len(prompt_to_ind)))
    need_to_denom[model] = np.zeros((len(metrics), len(prompt_to_ind)))
    for sys_prompt in sys_prompts:
        sys_0, sys_1 = sys_prompt.split('-')
        sys_0, sys_1 = int(sys_0), int(sys_1)
        for met_ind, metric in enumerate(metrics):
            if sys_0 != 0 and 'fair' in metric:
                continue
            if metric not in all_diffawares.keys():
                all_diffawares[metric] = np.zeros((len(models), len(prompt_to_ind), 3)) - 1
                all_ctxtawares[metric] = np.zeros((len(models), len(prompt_to_ind), 3)) - 1
                all_dists[metric] = np.zeros((len(models), len(prompt_to_ind), 2, 3)) - 1

            if metric == 'our-religion':
                datakey = 1000
                input_0, input_1 = 1000, 1001
            elif metric == 'our-occup-descript':
                datakey = 1002
                input_0, input_1 = 1002, 1003
            elif metric == 'our-legal':
                datakey = 1004
                input_0, input_1 = 1004, 1005
            elif metric == 'our-asylum':
                datakey = 1006
                input_0, input_1 = 1006, 1007 
            elif metric == 'our-bbq':
                datakey = 1008
                input_0, input_1 = 1008, 1009
            elif metric == 'our-sbf':
                datakey = 1010
                input_0, input_1 = 1010, 1011
            elif metric == 'our-occup-affirmact':
                datakey = 1012
                input_0, input_1 = 1012, 1013
            elif metric == 'our-cultapp':
                datakey = 1014
                input_0, input_1 = 1014, 1015
            elif metric == 'fair-bbqamb':
                datakey = 102
                input_0, input_1 = 102, 103
                if 'claude' in model or 'gpt' in model: # subset of bbq due to api costs
                    datakey = 100
                    input_0, input_1 = 100, 101
                elif '70b' in model: # due to size of model, split into two runs
                    input_0, input_1 = 102, 102
            elif metric == 'fair-bbqdis':
                datakey = 103
                input_0, input_1 = 102, 103
                if 'claude' in model or 'gpt' in model:  # subset of bbq due to api costs
                    datakey = 101
                    input_0, input_1 = 100, 101
                elif '70b' in model:
                    input_0, input_1 = 103, 103
            elif metric == 'fair-diseval':
                datakey = 200
                input_0, input_1 = 200, 200

            loc = 'in{0}-{1}_sys{2}-{3}_m{4}'.format(input_0, input_1, sys_0, sys_1, model)
            filename = utils.get_file(base+loc)
            if filename == '':
                continue
            print(filename)
            results = pickle.load(open(filename, 'rb'))[1]

            generated_prompts = utils.generate_input_prompts([datakey])
            steer = None
           
            # cleaning, and gives held_results for those with multiple, i.e., input_0 != input_1
            if 'fair' not in metric: # the shape is different
                if len(results) == 1:
                    to_bin = np.vectorize(utils.clean_response)
                    results = to_bin(np.array(results))

                    to_bin = np.vectorize(utils.map_to_abc)
                    results = to_bin(np.array(results))
                    for sys_in in range(len(sys_prompt_dict[sys_prompt])):
                        these_results = results[:, [sys_in]]
                        prompt_ind = prompt_to_ind[sys_prompt_dict[sys_prompt][sys_in]]
                        need_to_fix[model][met_ind][prompt_ind] += np.sum(these_results==-2)
                        need_to_ref[model][met_ind][prompt_ind] += np.sum(these_results==-1)
                        need_to_denom[model][met_ind][prompt_ind] += (np.sum(these_results!=-2)+np.sum(these_results==-2))
                    results[results==-2]=-1
                else:
                    to_bin_clean = np.vectorize(utils.clean_response)
                    to_bin_map = np.vectorize(utils.map_to_abc)
                    held_results = []
                    for rel_v in range(len(results)):
                        results_v = np.expand_dims(results[rel_v], 0)
                        results_v = to_bin_map(to_bin_clean(np.array(results_v)))
                        for sys_in in range(len(sys_prompt_dict[sys_prompt])):
                            these_results = results_v[:, [sys_in]]
                            prompt_ind = prompt_to_ind[sys_prompt_dict[sys_prompt][sys_in]]
                            need_to_fix[model][met_ind][prompt_ind] += np.sum(these_results==-2)
                            need_to_ref[model][met_ind][prompt_ind] += np.sum(these_results==-1)
                            need_to_denom[model][met_ind][prompt_ind] += (np.sum(these_results!=-2)+np.sum(these_results==-2))

                        results_v[results_v==-2]=-1
                        held_results.append(results_v)
            these_diffaware = []
            these_ctxtaware = []
            these_dists = []

            if 'our-' in metric:
                
                questions_diff, answers_diff, uniques_diff = generated_prompts[input_0][0], generated_prompts[input_0][1], generated_prompts[input_0][5]
                questions_eq, answers_eq, uniques_eq = generated_prompts[input_1][0], generated_prompts[input_1][1], generated_prompts[input_1][5]

                for sys_in in range(len(sys_prompt_dict[sys_prompt])):
                    diff_results = held_results[0][0][sys_in][:, 0]
                    equal_results = held_results[1][0][sys_in][:, 0]
                    diffaware, ctxtaware = analysis_utils.all_1k(diff_results, answers_diff, uniques_diff, equal_results, answers_eq, uniques_eq)
                   
                    these_diffaware.append(diffaware)
                    these_ctxtaware.append(ctxtaware)

                    diff_dist = [np.sum(diff_results==answers_diff), np.sum(diff_results==2)] # right, wrong, equal
                    wrong = len(diff_results)-np.sum(diff_results==-1)-np.sum(diff_dist)
                    diff_dist = [diff_dist[0], wrong, diff_dist[1]]

                    eq_dist = [0, 0, np.sum(equal_results==2)] # 0, wrong, equal
                    wrong = len(equal_results)-np.sum(equal_results==-1)-np.sum(eq_dist)
                    eq_dist[1] = wrong
                    these_dists.append([diff_dist, eq_dist])

            elif metric == 'fair-bbqamb':
                answers = generated_prompts[datakey][1]
                if '70b' not in model:
                    results = np.expand_dims(results[0], 0)

                to_bin = np.vectorize(utils.clean_response)
                results = to_bin(np.array(results))

                to_bin = np.vectorize(utils.map_to_abc)
                results = to_bin(results)


                for sys_in in range(len(sys_prompt_dict[sys_prompt])):
                    these_results = results[:, [sys_in]]
                    prompt_ind = prompt_to_ind[sys_prompt_dict[sys_prompt][sys_in]]
                    need_to_fix[model][met_ind][prompt_ind] += np.sum(these_results==-2)
                    need_to_ref[model][met_ind][prompt_ind] += np.sum(these_results==-1)
                    need_to_denom[model][met_ind][prompt_ind] += (np.sum(these_results!=-2)+np.sum(these_results==-2))
                results[results==-2]=-1

                assert np.mean(np.array(results.shape)[[0, 3]] == np.array([1, 1])) == 1

                for sys_in in range(len(sys_prompt_dict[sys_prompt])):
                    curr_mc = np.array(results[0][sys_in][:, 0]) 
                    this_val = analysis_utils.fair_bbqamb(curr_mc, answers)
                    these_diffaware.append(this_val)


            elif metric == 'fair-bbqdis':
                answers = generated_prompts[datakey][1]
                if '70b' not in model:
                    results = np.expand_dims(results[1], 0)

                to_bin = np.vectorize(utils.clean_response)
                results = to_bin(np.array(results))
                to_bin = np.vectorize(utils.map_to_abc)
                results = to_bin(results)

                for sys_in in range(len(sys_prompt_dict[sys_prompt])):
                    these_results = results[:, [sys_in]]
                    prompt_ind = prompt_to_ind[sys_prompt_dict[sys_prompt][sys_in]]
                    need_to_fix[model][met_ind][prompt_ind] += np.sum(these_results==-2)
                    need_to_ref[model][met_ind][prompt_ind] += np.sum(these_results==-1)
                    need_to_denom[model][met_ind][prompt_ind] += (np.sum(these_results!=-2)+np.sum(these_results==-2))

                results[results==-2]=-1

                assert np.mean(np.array(results.shape)[[0, 3]] == np.array([1, 1])) == 1
                for sys_in in range(len(sys_prompt_dict[sys_prompt])):
                    curr_mc = np.array(results[0][sys_in][:, 0]) 
                    this_val = analysis_utils.fair_bbqdis(curr_mc, answers)
                    these_diffaware.append(this_val)
            elif metric == 'fair-diseval':
                answers = generated_prompts[datakey][1]
                to_bin = np.vectorize(utils.clean_response)
                results = to_bin(results)
                to_bin = np.vectorize(utils.map_to_binary)
                results = to_bin(results)

                for sys_in in range(len(sys_prompt_dict[sys_prompt])):
                    these_results = results[:, [sys_in]]
                    prompt_ind = prompt_to_ind[sys_prompt_dict[sys_prompt][sys_in]]
                    need_to_fix[model][met_ind][prompt_ind] += np.sum(these_results==-2)
                    need_to_ref[model][met_ind][prompt_ind] += np.sum(these_results==-1)
                    need_to_denom[model][met_ind][prompt_ind] += (np.sum(these_results!=-2)+np.sum(these_results==-2))
                assert np.mean(np.array(results.shape)[[0, 3]] == np.array([1, 1])) == 1
                results[results==-2]=-1

                for sys_in in range(len(sys_prompt_dict[sys_prompt])):
                    curr_mc = np.array(results[0][sys_in][:, 0]) 
                    this_val = analysis_utils.fair_diseval(curr_mc, answers)
                    these_diffaware.append(this_val)

            for p_ind, prompt in enumerate(sys_prompt_dict[sys_prompt]):
                if 'fair' not in metric:
                    all_ctxtawares[metric][mod_ind][prompt_to_ind[prompt]] = these_ctxtaware[p_ind]
                    all_dists[metric][mod_ind][prompt_to_ind[prompt]] = these_dists[p_ind]
                all_diffawares[metric][mod_ind][prompt_to_ind[prompt]] = these_diffaware[p_ind]

pickle.dump([all_diffawares, all_ctxtawares, all_dists, models, metrics, prompt_to_ind, sys_prompt_dict, need_to_fix, need_to_ref, need_to_denom, sys_prompts], open('analyzed_{}.pkl'.format(file_name), 'wb')) 
print("Took {:.2f} minutes".format((time.time()-start)/60.))



