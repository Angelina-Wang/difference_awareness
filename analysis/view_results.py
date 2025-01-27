import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import pickle
import sys
import numpy as np
import analysis_utils
import math
from scipy.stats import pearsonr
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from sklearn.linear_model import RANSACRegressor, LinearRegression, TheilSenRegressor
from scipy.stats import theilslopes
from matplotlib.legend_handler import HandlerBase

sys.path.append('..')
import utils

#https://dovydas.com/blog/colorblind-friendly-diagrams# okabe and ito
color_palette = {'C0': '#E69F00', 'C1': '#56B4E9', 'C2': '#009E73', 'C3': '#F0E442', 'C4': '#0072B2', 'C5': '#D55E00', 'C6': '#CC79A7', 'C7': '#000000', 'C8': '.8', 'C9': '.6', 'C10': '.4'} # except for C8+ are shades of gray

file_name = 'all'
#file_name = 'templegal'
all_diffawares, all_ctxtawares, all_dists, models, metrics, prompt_to_ind, sys_prompt_dict, need_to_fix, need_to_ref, need_to_denom, sys_prompts = pickle.load(open('analyzed_{}.pkl'.format(file_name), 'rb')) 
assert len(models) == len(all_diffawares[list(all_diffawares.keys())[np.random.choice(np.arange(len(all_diffawares)))]])
assert len(models) == len(all_ctxtawares[list(all_ctxtawares.keys())[np.random.choice(np.arange(len(all_ctxtawares)))]])

sorted_metrics = ['fair-bbqamb', 'fair-bbqdis', 'fair-diseval', 'our-religion', 'our-occup-descript', 'our-legal', 'our-asylum', 'our-bbq', 'our-sbf', 'our-occup-affirmact', 'our-cultapp']
our_metrics = ['our-religion', 'our-occup-descript', 'our-legal', 'our-asylum', 'our-bbq', 'our-sbf', 'our-occup-affirmact', 'our-cultapp'] 
sorted_models = ['llama-2-7b', 'llama-3-8b', 'llama-3.1base-8b', 'llama-3.1uncensored-8b', 'llama-3.1-8b',  'gemma-1.1-7b', 'mistral-0.1-7b', 'mistral-0.2-7b', 'mistral-0.3base-7b', 'mistral-0.3uncensored-7b', 'mistral-0.3-7b', 'ministral-2410-8b', 'mistral-nemobase-12b', 'mistral-nemouncensored-12b', 'mistral-nemo-12b', 'gemma-2base-9b', 'gemma-2uncensored-9b', 'gemma-1-7b', 'gemma-1.1-7b', 'gemma-2-9b', 'gemma-2base-27b', 'gemma-2uncensored-27b', 'gemma-2-27b', 'llama-3.1-70b', 'claude-3.5-haiku', 'claude-3.5-sonnet', 'gpt-4o-mini', 'gpt-4o-']
historical =  ['llama-2-7b', 'llama-3-8b', 'gemma-1-7b', 'gemma-1.1-7b', 'mistral-0.2-7b', 'mistral-0.1-7b', 'ministral-2410-8b']
model_names_nice = {'gpt-4o-': 'GPT-4o', 'gpt-4o-mini': 'GPT-4o Mini', 'mistral-nemo-12b': 'Mistral 12b', 'mistral-0.3-7b': 'Mistral-0.3 7b', 'gemma-2-27b': 'Gemma-2 27b', 'gemma-2-9b': "Gemma-2 9b", 'llama-3.1-8b': 'Llama-3.1 8b', 'llama-3.1-70b': 'Llama-3.1 70b', 'claude-3.5-sonnet': 'Claude-3.5 Sonnet', 'claude-3.5-haiku': 'Claude-3.5 Haiku'}
metrics_ordered = sorted(metrics, key=lambda x: sorted_metrics.index(x))
models_order = sorted(models, key=lambda x: sorted_models.index(x))
it_models = utils.IT_MODELS
it_indices = [models.index(mod) for mod in it_models]
casestudy_models = utils.CASE_MODELS
bench_name = ['D1', 'D2', 'D3', 'D4', 'N1', 'N2', 'N3', "N4"]

metric_name = {'our-religion': 'D1', 'our-occup-descript': 'D2', 'our-legal': 'D3', 'our-asylum': 'D4', 'our-bbq': 'N1', 'our-sbf': 'N2', 'our-occup-affirmact': 'N3', 'our-cultapp': 'N4', 'fair-bbqamb': 'PW1', 'fair-bbqdis': 'PW2', 'fair-diseval': 'PW3'}

### Paper Figure 10: Overall performance for all models ###
fig, axes = plt.subplots(2, 1, figsize=(12, 8)) 
fontsize = 16
width = .08
for met_ind, metric in enumerate(our_metrics):

    for xtick, model in enumerate(it_models):
        mod_ind = models.index(model)
        diff = all_diffawares[metric][mod_ind][0]
        ctxt = all_ctxtawares[metric][mod_ind][0]

        if diff[1] != -1:
            axes[0].errorbar([met_ind+(xtick)*width], [diff[1]],yerr=[[diff[1]-diff[0]], [diff[2]-diff[1]]], c='k', ms=2, capsize=2)
            axes[0].bar([met_ind+(xtick)*width], [diff[1]], color=color_palette['C{}'.format(xtick//2)], width=width, alpha=1., hatch='' if xtick%2==0 else '/', edgecolor='k')
        if ctxt[1] != -1:
            axes[1].errorbar([met_ind+(xtick)*width], [ctxt[1]],yerr=[[ctxt[1]-ctxt[0]], [ctxt[2]-ctxt[1]]], c='k', ms=2, capsize=2)
            axes[1].bar([met_ind+(xtick)*width], [ctxt[1]], color=color_palette['C{}'.format(xtick//2)], width=width, alpha=1., hatch='' if xtick%2==0 else '/', edgecolor='k')
handles = []
for n, name in enumerate(it_models):
    handles.append(mpatches.Patch( facecolor=color_palette['C{}'.format(n//2)], hatch='' if n%2==0 else "//",label=model_names_nice[name], alpha=1.))
axes[0].set_ylim(-.1,1)
axes[1].set_ylim(-.1,1)
axes[1].legend(handles=handles, prop={'size': 8}, loc='lower right', ncol=5)
axes[0].set_title('DiffAware', fontsize=fontsize)
axes[1].set_title('CtxtAware', fontsize=fontsize)
axes[0].plot([-.1, 8.1], [.33, .33], c='k', linestyle='dashed')
axes[1].plot([-.1, 8.1], [.33, .33], c='k', linestyle='dashed')
axes[0].plot([3.9, len(our_metrics)], [0, 0], c='k', linestyle='dashed')
axes[1].plot([-.1, len(our_metrics)], [0, 0], c='k', linestyle='dashed')
axes[0].set_xticks(np.arange(len(our_metrics))+.38, [metric_name[chunk] for chunk in our_metrics], fontsize=fontsize)
axes[1].set_xticks(np.arange(len(our_metrics))+.38, [metric_name[chunk] for chunk in our_metrics], fontsize=fontsize)
axes[1].set_yticklabels(axes[1].get_yticks(), fontsize=fontsize)
axes[0].set_yticks([-.2, 0, .2, .4, .6, .8, 1], ['-.2', '0.', '.2', '.4', '.6', '.8', '1.'], fontsize=fontsize)
axes[1].set_yticks([-.2, 0, .2, .4, .6, .8, 1], ['-.2', '0.', '.2', '.4', '.6', '.8', '1.'], fontsize=fontsize)
plt.tight_layout()
plt.savefig('figures/10_overallmetric.png', dpi=300)
plt.close()

### Paper Figure 4: De-biasing prompts ### 
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fontsize = 14
for vers in range(2):
    ind_to_prompt = {v: k for k, v in prompt_to_ind.items()}
    linewidth=.9
    for m, model in enumerate(casestudy_models):
        mod_ind = models.index(model)
        axis_r = vers
        axis_c = m

        if axis_r == 0:
            axes[axis_r][axis_c].set_title(model.replace('gpt-4o-', 'GPT-4o').replace('mistral-nemo-12b', 'Mistral 12b').replace('gemma-2-27b', 'Gemma-2 27b').replace('llama-3.1-70b', 'Llama-3.1 70b').replace('claude-3.5-sonnet', 'Claude-3.5 Sonnet'), fontsize=fontsize, pad=10)

        for ind in range(len(prompt_to_ind)):
            if vers == 0:
                values = np.array([all_diffawares[met][mod_ind][ind] for met in our_metrics])
                ybase = .33
            else:
                values = np.array([all_ctxtawares[met][mod_ind][ind] for met in our_metrics])
                ybase = .33

            if ind_to_prompt[ind] in [1, 2, 3, 4]:
                color = 'C1'
            elif ind_to_prompt[ind] in [0]:
                color = 'C0'
            else:
                continue
            if color == 'C0':
                axes[axis_r][axis_c].bar(np.arange(4), values[:4, 1], color=color_palette[color])
                axes[axis_r][axis_c].bar(np.arange(4, 8), values[4:, 1], color=color_palette['C1'])
            else:
                axes[axis_r][axis_c].scatter(np.arange(8), values[:, 1], color='black', marker='_', s=155,  linewidth=1)
        for ind in range(len(bench_name)):
            axes[axis_r][axis_c].text(ind-.3, -.053, bench_name[ind], fontsize=6)
        axes[axis_r][axis_c].set_ylim(-.1, 1)
        axes[axis_r][axis_c].set_xticks([1.5, 5.5], ['Descrip', 'Norm'], fontsize=fontsize)
        axes[axis_r][axis_c].plot([-.6, 7.6], [ybase, ybase], linestyle='dashed', color='k', alpha=.5)
        axes[axis_r][axis_c].tick_params(axis='y', labelsize=fontsize)

class TwoColorHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        orange_square = Rectangle((xdescent, ydescent), width / 2.5, height,
                                  facecolor=color_palette['C0'], transform=trans)
        blue_square = Rectangle((xdescent + width / 2, ydescent), width / 2.5, height,
                                facecolor=color_palette['C1'], transform=trans)
        return [orange_square, blue_square]

custom_line = Line2D([0], [0], color='black', lw=1)
patch = mpatches.Patch( facecolor=color_palette['C0'])
axes[0][1].legend([Rectangle((0, 0), 1, 1), custom_line], ['Baseline','Debiasing'], fontsize=fontsize-3, handler_map={Rectangle: TwoColorHandler()})
axes[0][0].set_ylabel('DiffAware', fontsize=fontsize)
axes[1][0].set_ylabel('CtxtAware', fontsize=fontsize)
plt.tight_layout()
plt.savefig('figures/4_debiasingprompts.png', dpi=300)
plt.close()

### Figure not in paper: Mean Rate for our benchmark vs the fairness ones ### 
fig, axes = plt.subplots(2, 1, figsize=(8, 6)) 
fair_vals = []
fair_metrics = []
our_diffawares = []
our_ctxtawares = []
for fair_metric in metrics_ordered:
    if 'fair' in fair_metric:
        fair_vals.append([1-np.absolute(all_diffawares[fair_metric][mod_ind][0]) for mod_ind in it_indices])
        fair_metrics.append(fair_metric)
    else:
        our_diffawares.append([all_diffawares[fair_metric][mod_ind][0] for mod_ind in it_indices])
        our_ctxtawares.append([all_ctxtawares[fair_metric][mod_ind][0] for mod_ind in it_indices])

fair_vals = np.array(fair_vals) 
our_diffawares = np.array(our_diffawares)
our_ctxtawares = np.array(our_ctxtawares)
keep_indices = np.where(fair_vals[0][:, 1]!=0)[0]
remove_indices = np.where(fair_vals[0][:, 1]==0)[0]
fair_vals = fair_vals[:, keep_indices]
our_diffawares = our_diffawares[:, keep_indices]
our_ctxtawares = our_ctxtawares[:, keep_indices]

fontsize = 6
fair_winrate = [[] for _ in range(len(keep_indices))]
ourdiff_winrate = [[] for _ in range(len(keep_indices))]
ourctxt_winrate = [[] for _ in range(len(keep_indices))]

for m in range(len(keep_indices)):
    for met in range(len(fair_vals)):
        to_add = fair_vals[met, m, 1] > fair_vals[met, :, 1]
        to_add = np.delete(to_add, m)
        fair_winrate[m].extend(to_add)
    for met in range(len(our_diffawares)):
        to_add = our_diffawares[met, m, 1] > our_diffawares[met, :, 1]
        to_add = np.delete(to_add, m)
        ourdiff_winrate[m].extend(to_add)

        to_add = our_ctxtawares[met, m, 1] > our_ctxtawares[met, :, 1]
        to_add = np.delete(to_add, m)
        ourctxt_winrate[m].extend(to_add)

fair_winrate = [np.mean(chunk) for chunk in fair_winrate]
ourdiff_winrate = [np.mean(chunk) for chunk in ourdiff_winrate]
ourctxt_winrate = [np.mean(chunk) for chunk in ourctxt_winrate]

axes[0].scatter(fair_winrate, ourdiff_winrate)
axes[1].scatter(fair_winrate, ourctxt_winrate)
for m in range(len(keep_indices)):
    axes[0].text(fair_winrate[m], ourdiff_winrate[m], it_models[m])
    axes[1].text(fair_winrate[m], ourctxt_winrate[m], it_models[m])

axes[0].set_ylabel('Difference Awareness')
axes[0].set_xlabel('Existing Fairness')
axes[1].set_ylabel('Contextual Awareness')
axes[1].set_xlabel('Existing Fairness')

plt.tight_layout()
plt.savefig('figures/_fairvsours.png', dpi=300)
plt.close()

### Paper Figure 3: MMLU correlation with our benchmark suite ### 

## MMLU links, 5-shot no cot
# claude-3.5-sonnet: https://www.anthropic.com/news/claude-3-5-sonnet
# claude-3.5-haiku: https://assets.anthropic.com/m/1cd9d098ac3e6467/original/Claude-3-Model-Card-October-Addendum.pdf
# gpt-4o-: https://openai.com/index/hello-gpt-4o/
# gpt-4o-mini: https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/ik
# llama-3.1-8b and 70b: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
# gemma-2-9b and 27b: https://ai.google.dev/gemma/docs/model_card_2
# mistral-0.3-7b: https://mistral.ai/news/announcing-mistral-7b/ -- not the .3 version exactly, https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3 says 62.58 I will use this
# mistral-nemo-12b: https://build.nvidia.com/nv-mistralai/mistral-nemo-12b-instruct/modelcard

# 5-shot no cot
mmlus = {'llama-2-7b': 34.1, 'llama-3.1-8b': 69.4, 'llama-3.1-70b': 83.6, 'gemma-1.1-7b': 64.3, 'gemma-2-9b': 71.3, 'gemma-2-27b': 75.2, 'claude-3.5-sonnet': 88.7, 'gpt-4o-': 88.7, 'gpt-4o-mini': 82, 'claude-3.5-haiku': 77.6, 'mistral-nemo-12b': 68., 'mistral-0.3-7b': 62.58}

figsize=(8, 6)
fig, axes = plt.subplots(2, 1, figsize=figsize) 
cap_vals = [[mmlus[model] for model in it_models]]
cap_metrics = ['MMLU']
our_diffawares = []
our_ctxtawares = []
fair_vals = []
for fair_metric in metrics_ordered:
    if 'fair' in fair_metric:
        fair_vals.append([1-np.absolute(all_diffawares[fair_metric][mod_ind][0]) for mod_ind in it_indices])
    else:
        our_diffawares.append([all_diffawares[fair_metric][mod_ind][0] for mod_ind in it_indices])
        our_ctxtawares.append([all_ctxtawares[fair_metric][mod_ind][0] for mod_ind in it_indices])

cap_vals = np.array(cap_vals)
our_diffawares = np.array(our_diffawares)
our_ctxtawares = np.array(our_ctxtawares)
fair_vals = np.array(fair_vals)
keep_indices = np.where(cap_vals[0]!=0)[0]
remove_indices = np.where(cap_vals[0]==0)[0]
fair_vals = fair_vals[:, keep_indices]
our_diffawares = our_diffawares[:, keep_indices]
our_ctxtawares = our_ctxtawares[:, keep_indices]

fontsize = 6
cap_winrate = [[] for _ in range(len(keep_indices))]
ourdiff_winrate = [[] for _ in range(len(keep_indices))]
ourctxt_winrate = [[] for _ in range(len(keep_indices))]

for m in range(len(keep_indices)):
    for met in range(len(cap_vals)):
        to_add = cap_vals[met, m] > cap_vals[met, :]
        to_add = np.delete(to_add, m)
        cap_winrate[m].extend(to_add)
    for met in range(len(our_diffawares)):
        to_add = our_diffawares[met, m, 1] > our_diffawares[met, :, 1]
        to_add = np.delete(to_add, m)
        ourdiff_winrate[m].extend(to_add)

        to_add = our_ctxtawares[met, m, 1] > our_ctxtawares[met, :, 1]
        to_add = np.delete(to_add, m)
        ourctxt_winrate[m].extend(to_add)

cap_winrate = [np.mean(chunk) for chunk in cap_winrate]

cap_winrate = [mmlus[model] for model in it_models]
ourdiff_winrate = [np.mean(chunk) for chunk in ourdiff_winrate]
ourctxt_winrate = [np.mean(chunk) for chunk in ourctxt_winrate]

fontsize=14
axes[0].scatter(cap_winrate, ourdiff_winrate)
axes[1].scatter(cap_winrate, ourctxt_winrate)
for m in range(len(keep_indices)):
    if figsize == (8, 6):
        if it_models[m] == 'gpt-4o-':
            axes[0].text(cap_winrate[m]-2.1, ourdiff_winrate[m]+.04, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] == 'claude-3.5-sonnet':
            axes[0].text(cap_winrate[m]-6.5, ourdiff_winrate[m]-.1, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] == 'gemma-2-27b':
            axes[0].text(cap_winrate[m]-2, ourdiff_winrate[m]+.05, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] in ['llama-3.1-70b', 'gemma-2-9b', 'claude-3.5-haiku', 'mistral-nemo-12b', 'gpt-4o-mini']:
            pass
        elif it_models[m] == 'llama-3.1-8b':
            axes[0].text(cap_winrate[m]-4.1, ourdiff_winrate[m]+.04, model_names_nice[it_models[m]], fontsize=fontsize)
        else:
            axes[0].text(cap_winrate[m], ourdiff_winrate[m]+.04, model_names_nice[it_models[m]], fontsize=fontsize)

        if it_models[m] == 'gpt-4o-':
            axes[1].text(cap_winrate[m]-2.9, ourctxt_winrate[m]-.08, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] == 'claude-3.5-sonnet':
            axes[1].text(cap_winrate[m]-6.5, ourctxt_winrate[m]+.04, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] == 'gemma-2-27b':
            axes[1].text(cap_winrate[m]-1.9, ourctxt_winrate[m]+.04, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] in ['mistral-nemo-12b', 'gpt-4o-mini', 'gemma-2-9b', 'llama-3.1-70b', 'claude-3.5-haiku']:
            pass
        else:
            axes[1].text(cap_winrate[m], ourctxt_winrate[m]+.04, model_names_nice[it_models[m]], fontsize=fontsize)
    else:
        if it_models[m] == 'claude-3.5-haiku':
            axes[0].text(cap_winrate[m]-4, ourdiff_winrate[m]-.08, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] == 'gpt-4o-mini':
            axes[0].text(cap_winrate[m]-2.3, ourdiff_winrate[m]+.04, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] == 'gpt-4o-':
            axes[0].text(cap_winrate[m]-1.1, ourdiff_winrate[m]-.08, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] == 'claude-3.5-sonnet':
            axes[0].text(cap_winrate[m]-4.3, ourdiff_winrate[m]+.04, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] == 'llama-3.1-70b':
            axes[0].text(cap_winrate[m]-2., ourdiff_winrate[m]-.08, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] == 'llama-3.1-8b':
            axes[0].text(cap_winrate[m]-2.1, ourdiff_winrate[m]+.04, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] == 'gemma-2-9b':
            axes[0].text(cap_winrate[m]-1, ourdiff_winrate[m]+.04, model_names_nice[it_models[m]], fontsize=fontsize)
        else:
            axes[0].text(cap_winrate[m], ourdiff_winrate[m]+.04, model_names_nice[it_models[m]], fontsize=fontsize)

        if it_models[m] == 'gpt-4o-':
            axes[1].text(cap_winrate[m]-1.9, ourctxt_winrate[m]-.08, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] == 'claude-3.5-sonnet':
            axes[1].text(cap_winrate[m]-4.3, ourctxt_winrate[m]+.04, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] == 'gemma-2-9b':
            axes[1].text(cap_winrate[m]-1.9, ourctxt_winrate[m]+.04, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] == 'mistral-nemo-12b':
            axes[1].text(cap_winrate[m]-2.9, ourctxt_winrate[m]+.04, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] == 'gemma-2-27b':
            axes[1].text(cap_winrate[m]-2, ourctxt_winrate[m]+.04, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] == 'llama-3.1-70b':
            axes[1].text(cap_winrate[m]-4, ourctxt_winrate[m]+.04, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] == 'gpt-4o-mini':
            axes[1].text(cap_winrate[m]+.5, ourctxt_winrate[m]-.02, model_names_nice[it_models[m]], fontsize=fontsize)
        elif it_models[m] == 'claude-3.5-haiku':
            axes[1].text(cap_winrate[m]-.3, ourctxt_winrate[m]-.08, model_names_nice[it_models[m]], fontsize=fontsize)
        else:
            axes[1].text(cap_winrate[m], ourctxt_winrate[m]+.04, model_names_nice[it_models[m]], fontsize=fontsize)

pears = pearsonr(cap_winrate, ourdiff_winrate)
stat, pval = pears.statistic, pears.pvalue
axes[0].set_title("r={0:.2f}, p={1:.2f}".format(stat, pval), fontsize=fontsize)
pears = pearsonr(cap_winrate, ourctxt_winrate)
stat, pval = pears.statistic, pears.pvalue
if pval < .001:
    axes[1].set_title("r={0:.2f}, p<.001".format(stat), fontsize=fontsize)
else:
    axes[1].set_title("r={0:.2f}, p={1:.2f}".format(stat, pval), fontsize=fontsize)
axes[0].set_ylim(0, 1.)
axes[1].set_ylim(0, 1.)
axes[0].set_ylabel('DiffAware Winrate', fontsize=fontsize)
axes[0].set_xlabel('MMLU', fontsize=fontsize)
axes[1].set_ylabel('CtxtAware Winrate', fontsize=fontsize)
axes[1].set_xlabel('MMLU', fontsize=fontsize)

plt.tight_layout()
plt.savefig('figures/3_mmluvsours.png', dpi=300)
plt.close()

### Paper Figure 2: Fair models on our benchmark suite ### 
fig, axes = plt.subplots(2, 2, figsize=(9, 8)) 
fontsize=16
for mod_ind, model in enumerate(['gemma-2-27b', 'gpt-4o-mini', 'gemma-2-9b', 'gpt-4o-']):
    r, c = mod_ind //2, mod_ind%2
    axes[r][c].set_title(model.replace('gpt-4o-', 'GPT-4o').replace('gemma-2-9b', 'Gemma-2 9b'), fontsize=fontsize)

    metric_ordering = [[], [], []]
    for met_ind, metric in enumerate(metrics_ordered):
        if 'fair' in metric:
            val = 1-np.absolute(all_diffawares[metric][models.index(model)][0])
            metric_ordering[0].append(val[1])
        else:
            val = all_diffawares[metric][models.index(model)][0]
            if val[1] == -1:
                metric_ordering[1].append(0)
            else:
                metric_ordering[1].append(val[1])
            val = all_ctxtawares[metric][models.index(model)][0]
            if val[2] == -1:
                metric_ordering[2].append(0)
            else:
                metric_ordering[2].append(val[1])

    def get_rankings(lst):
        sorted_indices = sorted(range(len(lst)), key=lambda i: lst[i])
        rankings = [0] * len(lst)
        for rank, idx in enumerate(sorted_indices):
            rankings[idx] = rank + 1
        return rankings
    met_argsort = [get_rankings(metric_ordering[ind]) for ind in range(len(metric_ordering))]
    all_xticks = [0]*8
    for met_ind, metric in enumerate(metrics_ordered):
        if 'fair' in metric:
            val = 1-np.absolute(all_diffawares[metric][models.index(model)][0])
            min_val, max_val = np.amin(val), np.amax(val)
            x = met_argsort[0][met_ind]
            axes[r][c].bar(x, val[1], color=color_palette['C0'], width=.9, edgecolor='k', alpha=1.)
            axes[r][c].errorbar([x], [val[1]], yerr=[[val[1]-min_val], [max_val-val[1]]], c='k', ms=2, capsize=2)
        else:
            val = all_diffawares[metric][models.index(model)][0]
            if val[1] == -1:
                continue
            x = met_argsort[1][met_ind-3]
            axes[r][c].text(x+2.6, -.048, bench_name[met_ind-3], fontsize=8)
            axes[r][c].bar(x+3, val[1], color=color_palette['C1'], width=.9, edgecolor='k', alpha=1., hatch='/' if met_ind>6 else '')
            axes[r][c].errorbar([x+3], [val[1]], yerr=[[val[1]-val[0]], [val[2]-val[1]]], c='k', ms=2, capsize=2)

            val = all_ctxtawares[metric][models.index(model)][0]
            x = met_argsort[2][met_ind-3]
            axes[r][c].text(x+10.6, -.048, bench_name[met_ind-3], fontsize=8)

            axes[r][c].bar(x+11, val[1], color=color_palette['C2'], width=.9, hatch='/' if met_ind>6 else '', edgecolor='k', alpha=1.)
            axes[r][c].errorbar([x+11], [val[1]], yerr=[[val[1]-val[0]], [val[2]-val[1]]], c='k', ms=2, capsize=2)
    axes[r][c].set_xticks([2, 7.5, 15.5], ['Prior', 'DiffAware', 'CtxtAware'], fontsize=fontsize)
    axes[r][c].set_yticks([-.25, 0, .25, .5, .75, 1.], ['-.25', '0.', '.25', '.5', '.75', '1.'], fontsize=fontsize)
    axes[r][c].tick_params(axis='y', labelsize=fontsize)
    axes[r][c].plot([3.5, 19.5], [.33, .33], c='k', linestyle='dashed')
    axes[r][c].set_ylim(-.1, 1.)
    axes[r][c].set_xlim(0, 20)
handles = []
handles.append(mpatches.Patch( facecolor='w', hatch='//',label="Descrip", edgecolor='k'))
handles.append(mpatches.Patch( facecolor='w', label="Norm", edgecolor='k'))
axes[1][0].legend(handles=handles, prop={'size': fontsize-3}, ncol=2, loc='upper center')
plt.tight_layout()
plt.savefig('./figures/2_fairnessvsours.png', dpi=300)
plt.close()

### Paper Figure 7: Explicitly difference aware prompts ### 
fig, axes = plt.subplots(2, 5, figsize=(12, 4))
for vers in range(2):
    ind_to_prompt = {v: k for k, v in prompt_to_ind.items()}
    linewidth=.9
    for m, model in enumerate(casestudy_models):
        mod_ind = models.index(model)
        axis_r = vers
        axis_c = m

        if axis_r == 0:
            axes[axis_r][axis_c].set_title(model.replace('gpt-4o-', 'GPT-4o').replace('mistral-nemo-12b', 'Mistral 12b').replace('gemma-2-27b', 'Gemma-2 27b').replace('llama-3.1-70b', 'Llama-3.1 70b').replace('claude-3.5-sonnet', 'Claude-3.5 Sonnet'), fontsize=fontsize, pad=10)

        for ind in range(len(prompt_to_ind)):
            if vers == 0:
                values = np.array([all_diffawares[met][mod_ind][ind] for met in our_metrics])
                ybase = .33
            else:
                values = np.array([all_ctxtawares[met][mod_ind][ind] for met in our_metrics])
                ybase = .33

            if ind_to_prompt[ind] in [5, 6, 7]:
                color = 'C1'
            elif ind_to_prompt[ind] in [0]:
                color = 'C0'
            else:
                continue
            if color == 'C0':
                axes[axis_r][axis_c].bar(np.arange(4), values[:4, 1], color=color_palette[color])
                axes[axis_r][axis_c].bar(np.arange(4, 8), values[4:, 1], color=color_palette['C1'])
            else:
                axes[axis_r][axis_c].scatter(np.arange(8), values[:, 1], color='black', marker='_', s=155,  linewidth=1)

        axes[axis_r][axis_c].set_ylim(-.1, 1)
        axes[axis_r][axis_c].plot([-.6, 7.6], [ybase, ybase], linestyle='dashed', color='k', alpha=.5)
        axes[axis_r][axis_c].set_xticks([1.5, 5.5], ['Descrip', 'Norm'], fontsize=fontsize)

custom_line = Line2D([0], [0], color='black', lw=1)
patch = mpatches.Patch( facecolor=color_palette['C0'])
axes[0][1].legend([Rectangle((0, 0), 1, 1), custom_line], ['Baseline','Prompt'], fontsize=fontsize-5.5, handler_map={Rectangle: TwoColorHandler()})
axes[0][0].set_ylabel('DiffAware', fontsize=fontsize)
axes[1][0].set_ylabel('CtxtAware', fontsize=fontsize)
plt.tight_layout()
plt.savefig('figures/7_diffawareprompts.png', dpi=300)
plt.close()

### Paper Figure 9: benchmark correlation ###
fig, axes = plt.subplots(2, 1, figsize=(8, 8)) 
benchmark_names = [f"Benchmark {i+1}" for i in range(8)]
benchmark_names = [metric_name[chunk] for chunk in sorted_metrics]
model_names = [f"Model {i+1}" for i in range(10)]
data_diff = np.zeros((11, 11))
data_ctxt = np.zeros((11, 11)) 
for met_ind_1, metric_1 in enumerate(sorted_metrics):
    for met_ind_2, metric_2 in enumerate(sorted_metrics):

        diff_1 = [all_diffawares[metric_1][models.index(model)][0][1] for model in it_models]
        diff_2 = [all_diffawares[metric_2][models.index(model)][0][1] for model in it_models]
        ctxt_1 = [all_ctxtawares[metric_1][models.index(model)][0][1] for model in it_models]
        ctxt_2 = [all_ctxtawares[metric_2][models.index(model)][0][1] for model in it_models]

        if 'fair' in metric_1:
            diff_1 = [1-np.absolute(val) for val in diff_1]
        if 'fair' in metric_2:
            diff_2 = [1-np.absolute(val) for val in diff_2]
        
        pears = pearsonr(diff_1, diff_2)
        stat, pval = pears.statistic, pears.pvalue
        data_diff[met_ind_1][met_ind_2] = stat

        pears = pearsonr(ctxt_1, ctxt_2)
        stat, pval = pears.statistic, pears.pvalue
        data_ctxt[met_ind_1][met_ind_2] = stat

df_diff = pd.DataFrame(data_diff, index=benchmark_names, columns=benchmark_names)
df_ctxt = pd.DataFrame(data_ctxt, index=benchmark_names, columns=benchmark_names)

mask = np.zeros_like(df_diff, dtype='bool')
mask[np.triu_indices_from(mask)] = True

for i in range(2):
    ax = sns.heatmap(df_diff if i == 0 else df_ctxt, mask=mask, annot=True, cmap="coolwarm_r", cbar_kws={'label': 'DiffAware' if i == 0 else 'CtxtAware'}, ax=axes[i], center=0)
    highlight_cells = [(4, 3), (8, 7)]  # (row, column) pairs to highlight
    for row, col in highlight_cells:
        rect = Rectangle((col, row), 3, 3, fill=False, edgecolor='black', lw=2)
        ax.add_patch(rect)

plt.tight_layout()
plt.savefig('figures/9_benchmarkcorrelations.png', dpi=300)
plt.close()

### Paper Figure 8: Refusals 
fig, axes = plt.subplots(3, 4, figsize=(12, 8))
fontsize = 16
for met_ind, metric in enumerate(metrics_ordered):
    if met_ind == 0:
        xticks = []
    if 'fair' in metric:
        ax_r = 0
        ax_c = met_ind
    else:
        if metric in ['our-occup-descript', 'our-religion', 'our-legal', 'our-asylum']:
            ax_r = 1
            ax_c = met_ind-3
        else:
            ax_r = 2
            ax_c = met_ind-7
    axes[ax_r][ax_c].set_title(metric_name[metric], fontsize=fontsize)

    for mod_ind, model in enumerate(it_models):
        re_mod_ind = models.index(model)
        fix_perc = need_to_fix[model][metrics.index(metric)][0]/need_to_denom[model][metrics.index(metric)][0]
        ref_perc = need_to_ref[model][metrics.index(metric)][0]/need_to_denom[model][metrics.index(metric)][0]

        axes[ax_r][ax_c].bar(mod_ind, ref_perc, color=color_palette['C0'], width=.8)
        axes[ax_r][ax_c].bar(mod_ind, fix_perc, bottom=ref_perc, color=color_palette['C1'], width=.8)

for ax_r in range(len(axes)):
    for ax_c in range(len(axes[ax_r])):
        axes[ax_r][ax_c].set_xticks(np.arange(len(it_models)), [model_names_nice[mod] for mod in it_models], rotation=90, fontsize=fontsize-5)
        axes[ax_r][ax_c].set_ylim(0, 1.)
        axes[ax_r][ax_c].set_yticks([0, .2, .4, .6, .8, 1.], ['0', '20', '40', '60', '80', '100'], fontsize=fontsize-5)

axes[2][2].scatter([], [], c='C0', label='Refusal %')
axes[2][2].scatter([], [], c='C1', label='Unknown %')
hatches = ['/', '+', '*']
labels = ['% Refusal', '% Invalid']
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color_palette['C0']), plt.Rectangle((0, 0), 1, 1, facecolor=color_palette['C1'])]

axes[0][3].axis('off')
axes[0][3].legend(handles, labels, prop={'size': fontsize})

axes[0][0].set_ylabel('Prior Work', fontsize=fontsize)
axes[1][0].set_ylabel('Descriptive', fontsize=fontsize)
axes[2][0].set_ylabel('Normative', fontsize=fontsize)
plt.tight_layout()
plt.savefig('figures/8_refusals.png', dpi=300)
plt.close()





