import copy
import os
import time

import pandas as pd
import numpy as np
from tqdm import tqdm


def generate_counterfactuals(methods, method_names, facts, outcome, env, eval_path, params):
    ''' Generates counterfactual explanations for each passed failure trajectory using each model '''
    print('Generating counterfactuals for {} facts'.format(len(facts)))
    for i_m, m in enumerate(methods):
        start = time.time()
        record = []
        eval_path_results = os.path.join(eval_path, f'{method_names[i_m]}/{outcome.name}_results.csv')
        print('Method = {}'.format(method_names[i_m]))
        for i, t in tqdm(enumerate(facts)):
            res = m.generate_counterfactuals(t)
            for cf in res:
                record.append([i,
                               list(t.states[0]),
                               list(t.end_state),
                               list(t.actions),
                               list(cf.recourse),
                               *list(cf.reward_dict.values()),
                               cf.value])

            columns = ['Fact id',
                       'Fact start state',
                       'Fact end state',
                       'Fact actions',
                       'Recourse'] + m.obj.objectives + m.obj.constraints + ['Value']
            df = pd.DataFrame(record, columns=columns)
            df.to_csv(eval_path_results)

        end = time.time()
        print('Method = {} Task = {} Average time for one counterfactual = {}'.format(method_names[i_m],
                                                                                      params['task_name'],
                                                                                      (end-start)/len(facts)))


def evaluate_coverage(methods, method_names, eval_path, total_facts, params):
    printout = '{: ^20}'.format('Algorithm') + '|' +\
               '{: ^20}'.format('Number facts') + '|' +\
               '{: ^20}'.format('Generated cfs (%)') + '|' + '\n'
    printout += '-' * ((25+1)*3) + '\n'
    for i_m, m in enumerate(methods):
        eval_path_results = os.path.join(eval_path, f'{method_names[i_m]}/results.csv')
        df = pd.read_csv(eval_path_results, header=0)
        facts = pd.unique(df['Fact id'])

        coverage = (1.0*len(facts)) / total_facts
        printout += '{: ^20}'.format(method_names[i_m]) + '|' +\
                    '{: ^20}'.format(total_facts) + '|' +\
                    '{: ^20.2}'.format(coverage) + '|' '\n'

    print(printout)


def evaluate_cf_properties(methods, method_names, eval_path, params):
    objectives = methods[0].obj.objectives + methods[0].obj.constraints  # assuming all methods evaluated on the same metrics
    printout = '{: ^25}'.format('Algorithm/Metric') + '|' + \
               ''.join(['{: ^12}'.format(obj_name) + '|' for obj_name in objectives]) + '\n'
    printout += '-' * (25 + (len(objectives)+1)*12) + '\n'

    for i_m, m in enumerate(methods):
        eval_path_results = os.path.join(eval_path, f'{method_names[i_m]}/results.csv')
        eval = {}
        df = pd.read_csv(eval_path_results, header=0)

        for obj_name in objectives:
            df_filtered = copy.copy(df)
            if obj_name not in m.obj.constraints:
                # filter only those where validity (or any other constraints) are satisfied
                for c in m.obj.constraints:
                    df_filtered = df_filtered[df_filtered[c] == 0]

            avg_val = np.mean(df_filtered[obj_name].values)
            eval[obj_name] = avg_val

        printout += '{: ^25}'.format(method_names[i_m]) + '|'
        for eval_key, eval_val in eval.items():
            printout += '{:^12.4}'.format(eval_val) + '|'

        printout += '\n'

    print(printout)


def evaluate_diversity(methods, method_names, eval_path, params):
    printout = '{: ^20}'.format('Algorithm') + '|' + \
               '{: ^20}'.format('Quantity') + '|' + \
               '{: ^20}'.format('Metric diversity') + '|' + \
               '{: ^20}'.format('Action diversity') + '|' + '\n'
    printout += '-' * ((20 + 1) * 4) + '\n'

    for i_m, m in enumerate(methods):
        eval_path_results = os.path.join(eval_path, f'{method_names[i_m]}/results.csv')
        df = pd.read_csv(eval_path_results, header=0)

        q = evaluate_quantity(df, method_names[i_m])
        metric_div = evaluate_metric_diversity(df, method_names[i_m], m.obj)
        action_div = evaluate_action_diversity(df, method_names[i_m], m.obj)

        printout += '{: ^20}'.format(method_names[i_m]) + '|' + \
                    '{: ^20.4}'.format(q) + '|' + \
                    '{: ^20.4}'.format(metric_div) + '|' + \
                    '{: ^20.4}'.format(action_div) + '|' + '\n'

    print(printout)

def evaluate_quantity(df, method_name=''):
    facts = pd.unique(df['Fact id'])

    cfs = []
    for f in facts:
        n = len(df[df['Fact id'] == f])
        cfs.append(n)

    return np.mean(cfs)

def evaluate_metric_diversity(df, method_name, obj):
    facts = pd.unique(df['Fact id'])
    metrics = obj.objectives
    diversity = []

    for f in facts:
        df_fact = df[df['Fact id'] == f]
        for i, x in df_fact.iterrows():
            for j, y in df_fact.iterrows():
                if i != j:
                    diff = 0
                    for m in metrics:
                        diff += (x[m] - y[m]) ** 2

                    diversity.append(diff)

    avg_div = np.mean(diversity)

    return avg_div

def evaluate_action_diversity(df, method_name, obj):
    facts = pd.unique(df['Fact id'])
    diversity = []

    for f in facts:
        df_fact = df[df['Fact id'] == f]
        for i, x in df_fact.iterrows():
            for j, y in df_fact.iterrows():
                if i != j:
                    diff = obj.action_proximity(x['Recourse'], y['Recourse'])
                    diversity.append(diff)

    avg_div = np.mean(diversity)
    return avg_div