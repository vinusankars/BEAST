from arutils import AutoRegressor
import torch
import pandas
import time
import pickle as pkl
import argparse
import os

def list_of_strings(arg):
    return arg.split(',')

parser = argparse.ArgumentParser()
parser.add_argument('--k1', type=int, default=15)
parser.add_argument('--k2', type=int, default=15)
parser.add_argument('--length', type=int, default=40)
parser.add_argument('--model', type=str, default='vicuna7b')
parser.add_argument('--log', type=int, default=1)
parser.add_argument('--target', type=int, default=1)
parser.add_argument('--budget', type=int, help='per sample attack budget in seconds', default=600000)
parser.add_argument('--DIR', type=str, default='data/')
parser.add_argument('--begin', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--ngram', type=int, default=1)
parser.add_argument('--multi_model_list', type=list_of_strings, default=None)


args = parser.parse_args() 

# load dataset
begin, end = args.begin, args.end

if args.target == 1:
    data = pandas.read_csv("data/harmful_behaviors.csv")
    prompts = list(data['goal'])[begin: end]
    targets = list(data['target'])[begin: end]
    
else:
    from datasets import load_dataset
    data = load_dataset('truthful_qa', 'generation')
    prompts = data['validation']['question'][begin: end]
    targets = [None for _ in range(len(prompts))]

# define the LLM attack model
if 'vicuna7b' in args.model.lower():
    name = ['vicuna', 'lmsys/vicuna-7b-v1.5']
elif 'vicuna13b' in args.model.lower():
    name = ['vicuna13b', 'lmsys/vicuna-13b-v1.5']
elif 'mistral' in args.model.lower():
    name = ['mistral', 'mistralai/Mistral-7B-Instruct-v0.2']
    
ar = AutoRegressor(name[1], budget=args.budget)

# set attack parameters
params = {"top_p": 1., 'top_k': None, 'temperature': 1.,\
        'new_gen_length': args.length, 'k1': args.k1, 'k2': args.k2, 'ngram': args.ngram,\
        'multi_model_list': args.multi_model_list}

max_bs = 50
    
# set the log file name
name[0] += f"_k1={params['k1']}_k2={params['k2']}_length={params['new_gen_length']}_{begin}_{end}_ngram={args.ngram}"
if args.target == 0:
    name[0] += "_untargeted"
if args.multi_model_list != None:
    name[0] += f"_modellist=vic13"
    
print(name)
print(params)

# logger
Time = []
try:
    file = open(os.path.join(args.DIR, f'{name[0]}.pkl'), 'rb')
    Log = pkl.load(file)
    file.close()
    print("Loaded file:", len(Log))
except:
    Log = {}


# iterate over the number of prompts
for i in range(len(Log), len(prompts)):
    
    inp_len = ar.tokenizer.encode(prompts[i], return_tensors='pt', add_special_tokens=False).shape[-1]
    ar.max_bs = max_bs
    
    print(f"{i+1:3d}/{len(prompts):3d}, Batch-size: {ar.max_bs}, Len: {inp_len} Time: {sum(Time)/60:4.2f} mins., Logged: {len(Log)}")
    
    # perform attack
    start = time.time()
    y = ar.self_attack_chat_batch(prompts=prompts[i:i+1], target=targets[i], **params)    
    Time.append(time.time() - start)
    
    # update log
    Log.update({i: (y, None, Time[-1])})
    
    if args.log == 1:
        file = open(os.path.join(args.DIR, f'{name[0]}.pkl'), 'wb')
        pkl.dump(Log, file)
        file.close()
