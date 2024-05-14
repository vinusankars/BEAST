import torch
import numpy as np
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import time
from concurrent.futures import ThreadPoolExecutor

class AutoRegressor():
    
    @torch.no_grad()
    def __init__(self, name="lmsys/vicuna-7b-v1.5", budget=None):
        
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, device_map="auto", torch_dtype=torch.float16, use_cache=False, low_cpu_mem_usage=True)
        self.multimodel = None
        
        # keep do_sample = False when temperature is None
        self.model.generation_config.do_sample = True if (self.model.generation_config.temperature != None) else False
        self.chat_format = ChatFormat(name)   
        self.sample = sample_top_p
        self.budget = budget
        
        # Correct error in Vicuna HF since LLaMA corrected it
        # They had swapped the values for top_p and temperature
        if "vicuna" in name.lower(): 
            self.model.generation_config.top_p = 0.9
            self.model.generation_config.temperature = 0.6
        
        if self.tokenizer.pad_token is None:   
            self.tokenizer.pad_token = "[PAD]"
            
            
    
    @torch.no_grad()
    def generate_n_tokens_batch(self, prompt_tokens, max_gen_len, temperature=None, top_p=None, top_k=None):
        """
        Generates max_gen_len number of tokens that are from prompt_len to total_len
        
        Params:
            prompt_tokens: batch of token ids, should be stackable
            max_gen_len: generate max_gen_len more tokens
            temperature: for generation of tokens
            top_p: for generation sampling
            top_k: for generation sampling
            
        Returns:
            logits: stacked logit value sequences (in batch) for generated token positions
            tokens: output token id sequences in batch
        """
                    
        assert max(len(i) for i in prompt_tokens) == min(len(i) for i in prompt_tokens), "Need to pad the batch"
        if not isinstance(prompt_tokens, torch.Tensor):
            prompt_tokens = torch.as_tensor(prompt_tokens)
            
        if max_gen_len == 0:
            return None, prompt_tokens
        
        config = copy.deepcopy(self.model.generation_config)
        if temperature != None: self.model.generation_config.temperature = temperature
        if top_p != None: self.model.generation_config.top_p = top_p
        if top_k != None: self.model.generation_config.top_k = top_k
        
        prompt_len = len(prompt_tokens[0])
        kwargs = copy.deepcopy({"generation_config": self.model.generation_config, "max_length": max_gen_len+prompt_len, \
            "min_length": max_gen_len+prompt_len, "return_dict_in_generate": True, "output_scores": True})
        out = self.model.generate(prompt_tokens.to(0), **kwargs)
        
        tokens = out.sequences
        logits = torch.stack(out.scores)
        logits = torch.permute(logits, (1, 0, 2))
        
        self.model.generation_config = config # reset model config
        return logits, tokens
    
    @torch.no_grad()
    def self_attack_chat_batch(self, prompts, k1=15, k2=15, lookahead_length=10, n_trials=1, top_p=None, top_k=None, temperature=None,\
        new_gen_length=10, target=None, verbose=1, interactive=0, ngram=1, multi_model_list=None):
        
        """
        Self attack to generate adversarial input prompts, where the first few tokens are given `prompt` and the rest
        `new_gen_length` tokens are generated via beam search
        
        Params: 
            prompts: Input prompt
            k1: no. of candidates in beam 
            k2: no. of candidates per candidate evaluated 
            lookahead_length: number of tokens in the output sequence to be considered in attack objective
            n_trials: number of generations over which the attack objective should be computed
            top_p: model generation hyperparameter for sampling
            top_k: model generation hyperparameter for sampling
            temperature: model generation hyperparameter for sampling
            new_gen_length: number of new adversarial tokens to be generated
            verbose: Bool to print verbose
            target: target phrase for targeted attack
            interactive: print scores and prompts for each step if value equals 1
            ngram: no. of tokens to add at each attack iteration; only 1 token will be adversarially sampled
            multi_model_list: evaluate objective using more models
        
        Returns:
            prompt_tokens: the potential adversarial prompts
            scores: corresponding attack objectives achieved for each `prompt_tokens` element
        """
        
        # intiialize model settings
        config = copy.deepcopy(self.model.generation_config)
        self.model.generation_config.temperature = temperature
        self.model.generation_config.top_p = top_p
        self.model.generation_config.top_k = top_k
        if self.model.generation_config.temperature != None:
            self.model.generation_config.do_sample = True 
        else:
            self.model.generation_config.do_sample = False
        
        system = ("{}{}{}".format(*self.chat_format.system)) if (self.chat_format.system[1] != "") else ""
        begin_inst_token = self.tokenizer.encode(self.chat_format.sep[0] + system + self.chat_format.user[0], add_special_tokens=False)
        end_inst_token = self.tokenizer.encode(self.chat_format.user[1] + self.chat_format.assistant[0], add_special_tokens=False)
        
        max_bs, bs = self.max_bs, len(prompts) 
        prompt_tokens = []
        

        if multi_model_list != None:
            if self.multimodel == None:
                self.multimodel = MultiModel(self, multi_model_list)
        
        # assume only 1 round of user-assistant dialog
        # tokenizer prompts
        for prompt in prompts: 
            prompt_tokens.append(self.tokenizer.encode(self.chat_format.sep[0] + system + self.chat_format.user[0] + prompt.strip(" "), add_special_tokens=False))

        logits, prompt_length = [], len(prompt_tokens[0])
        for b in range(0, len(prompt_tokens), max_bs):
            logits.append(self.generate_n_tokens_batch(prompt_tokens[b: b+max_bs], max_gen_len=1, temperature=temperature, top_p=top_p, top_k=top_k)[0])
        logits = torch.cat(logits, dim=0)

        # `curr_tokens` maintains a list[list[list]], (# batch_size) x (# beam candidates) x (# tokens) 
        curr_tokens = self.sample(torch.softmax(logits[:, 0], dim=-1), top_p, return_tokens=k1)[:, : k1]
        curr_tokens = [[i2 + [j.cpu().numpy().tolist()] for j in i1] for (i1, i2) in zip(curr_tokens, prompt_tokens)]
        
        start_time = time.time()
        best_scores, best_prompts = (np.zeros(np.stack(curr_tokens).shape[:2]) - np.inf).tolist(), np.zeros(np.stack(curr_tokens).shape[:2]).tolist()
        
        for l in range((new_gen_length-1) * ngram):
            
            if self.budget != None and (time.time() - start_time) > self.budget:
                break
            
            if verbose == 1: print(f"{l+2:3d}/{new_gen_length * ngram:3d}, Time: {(time.time()-start_time)/60:3.2f}, Size: {len(curr_tokens[0][0])}")            
            curr_tokens_ = [item for sublist in curr_tokens for item in sublist] # (# batch_size x # beam candidates) x (# tokens) 

            next_tokens = []
            for b in range(0, len(curr_tokens_), max_bs):
                logits = self.generate_n_tokens_batch(curr_tokens_[b: b+max_bs], max_gen_len=1, temperature=temperature, top_p=top_p, top_k=top_k)[0]
                next_tokens_ = self.sample(torch.softmax(logits[:, 0], dim=-1) , top_p, return_tokens=k2)[:, : k2]
                next_tokens.extend([[j.cpu().numpy().tolist() for j in i] for i in next_tokens_])

            score_prompt_tokens = []
            for i in range(len(curr_tokens_)):
                for j in range(len(next_tokens[i])):
                    score_prompt_tokens.append(curr_tokens_[i] + [next_tokens[i][j]]) 
            
            if l % ngram != 0:
                curr_tokens = copy.deepcopy([[score_prompt_tokens[ii] for ii in range(0, len(score_prompt_tokens), k1)]])
                continue
                
            scores = np.zeros(len(score_prompt_tokens))

            for b in range(0, len(score_prompt_tokens), max_bs):
                for _ in range(n_trials):
                                       
                    all_tokens = copy.deepcopy(score_prompt_tokens[b: b+max_bs]) # (# bacth_size) x (# total tokens)                    
                    for i in range(len(all_tokens)):
                        # concatenate the [/INST] tag before generating LLM output                        
                        all_tokens[i] = all_tokens[i] + end_inst_token
                    
                    # targeted attack for jailbreaks
                    if target is not None:
                        
                        def thread1():
                            return self.attack_objective_targeted(all_tokens, copy.deepcopy(target))
                        def thread2():
                            return self.multimodel.get_targetted_scores(all_tokens, copy.deepcopy(target))
                        
                        with ThreadPoolExecutor() as executor:
                            result1 = executor.submit(thread1)
                            if multi_model_list != None:
                                result2 = executor.submit(thread2)
                            scores[b: b+max_bs] += result1.result()
                            if multi_model_list != None:
                                scores[b: b+max_bs] += result2.result()
                        
                    # untargeted attacks for hallucinations
                    else:                  
                        toks = self.generate_n_tokens_batch(all_tokens, lookahead_length, temperature=temperature, top_p=top_p, top_k=top_k)[1]
                        scores[b: b+max_bs] += self.attack_objective_untargeted(toks, lookahead_length)

                scores[b: b+max_bs] = scores[b: b+max_bs] / n_trials
                
            curr_tokens_, scores_ = [], []      
                        
            for i in range(0, len(scores), len(scores)//bs):
                idx = i + np.argsort(scores[i: i+len(scores)//bs])[-k1: ]
                curr_tokens_.append([score_prompt_tokens[j] for j in idx])
                scores_.append(scores[idx].tolist())
                
            curr_tokens, scores = copy.deepcopy(curr_tokens_), copy.deepcopy(scores_)
            
            if interactive == 1:
                print("Scores:", np.stack(scores)[:, -1])
                print("Clean prompt:", self.tokenizer.decode(curr_tokens[0][-1][max(0, len(begin_inst_token)-1): prompt_length], skip_special_tokens=True))
                print("Adversarial suffix:", self.tokenizer.decode(curr_tokens[0][-1][prompt_length: ], skip_special_tokens=True))
                print()
                
            # Keep best prompts           
            best_kept = ""
            for i in range(len(scores)):
                old_best = np.max(best_scores[i])
                best_scores[i] += copy.deepcopy(scores[i])
                best_prompts[i] += copy.deepcopy(curr_tokens[i])
                ind = np.argsort(best_scores[i])[-k1: ]
                best_scores[i] = [best_scores[i][j] for j in ind]
                best_prompts[i] = [best_prompts[i][j] for j in ind]
                if old_best <= np.max(scores[i]): best_kept = "<Best kept>"

            if verbose == 1 and interactive == 0:
                print("Scores:", np.stack(scores)[:, -1], best_kept)                        
    
        # reset to default model settings
        self.model.generation_config = config
        for i in range(len(curr_tokens)): 
            for j in range(len(curr_tokens[i])): 
                curr_tokens[i][j] = curr_tokens[i][j][max(0, len(begin_inst_token)-1): ]
                best_prompts[i][j] = best_prompts[i][j][max(0, len(begin_inst_token)-1): ]
            
        return curr_tokens, scores, best_prompts, best_scores
    
    
    @torch.no_grad()
    def attack_objective_untargeted(self, toks, look_ahead_length):
        scores = self.perplexity(toks[:, :-look_ahead_length], toks)
        return scores.cpu().detach().numpy() 
      
    @torch.no_grad()
    def attack_objective_targeted(self, tokens, target):
        # tokens are "prompt tokens + adv tokens + [/INST]"
        if not isinstance(tokens, torch.Tensor): tokens = torch.tensor(tokens) 
        if tokens.device.type != 'cuda': tokens = tokens.to(0)
        scores = np.zeros(len(tokens))
            
        target = [self.tokenizer.encode(target, return_tensors='pt', add_special_tokens=False).to(0)]
        for i, t in enumerate(target):
            tokens_ = []
            for ii in range(len(tokens)):
                tokens_.append(torch.cat([tokens[ii:ii+1], t], dim=1))
            tokens_ = torch.cat(tokens_, dim=0).type(tokens.dtype)
            if tokens.shape == tokens_.shape:
                bos = self.tokenizer.encode(self.tokenizer.bos_token, return_tensors='pt', add_special_tokens=False).to(0)
                bos = torch.cat([bos]*len(tokens_), dim=0)
                tokens_ = torch.cat([bos, tokens_], dim=1).type(tokens_.dtype)
                scores += - np.stack(self.perplexity(tokens_[:, :1], tokens_).detach().cpu().numpy()) 
            else:
                scores += - np.stack(self.perplexity(tokens, tokens_).detach().cpu().numpy()) 

        scores = scores / len(target) 
        return scores      
    
    
    @torch.no_grad()
    def perplexity(self, x1, x2, return_logits=False):
        
        kwargs = {'input_ids': x2, 'use_cache': False, 'past_key_values': None, 'output_attentions': True, 'output_hidden_states': True, 'return_dict': True}
        output = self.model(**kwargs)
        softmax = torch.nn.Softmax(dim=-1)
        logs = None
        
        for curr_pos in range(len(x1[0]), len(x2[0])):            
            log = - torch.log(softmax(output.logits)[torch.arange(len(output.logits)), curr_pos-1, x2[:, curr_pos]])
            
            try: logs += log
            except: logs = log
           
        if return_logits:
            return torch.exp(logs / (len(x2[0]) - len(x1[0]))), output.logits
        else: 
            return torch.exp(logs / (len(x2[0]) - len(x1[0])))
    
class ChatFormat():
    
    def __init__(self, name):
        
        self.name = name.split('/')[-1]
        
        if "vicuna" in name.lower():
            self.system = ["", \
                "A chat between a curious user and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the user's questions. ", \
                ""]
            self.user = ["USER: ", ""]
            self.assistant = [" ASSISTANT:", ""]
            self.sep = ["", ""]
            
        elif "mistral" in name.lower():
            self.system = ["<<SYS>>\n", "", "\n<</SYS>>\n\n"]
            self.user = ["[INST] ", " [/INST]"]
            self.assistant = ["", ""]
            self.sep = ["<s>", "</s>"]
            
    def prepare_input(self, sens):        
        # assert only one user-assistant dialog
        system = "{}{}{}".format(*self.system) if (self.system[1] != "") else ""
        x = copy.deepcopy(sens)
        for i in range(len(x)):
            if "vicuna" in self.name.lower():
                x[i] = "{}{}{}{}{}".format(self.sep[0], system, self.user[0], x[i].strip(" "), self.assistant[0])
            elif "mistral" in self.name.lower():
                x[i] = "{}{}{}{}{}".format(self.sep[0], self.user[0], system, x[i].strip(" "), self.user[1])
        return x
    
    def get_prompt(self, sens): # reverse of prepare_input
        system = "{}{}{}".format(*self.system) if (self.system[1] != "") else ""
        x = copy.deepcopy(sens)
        for i in range(len(x)):
            x[i] = x[i].split(self.user[0])[1]                
            if "vicuna" in self.name.lower():
                x[i] = x[i].split(self.assistant[0])[0].strip()
            elif "mistral" in self.name.lower():
                if system != "":
                    x[i] = x[i].split(system)[1]
                x[i] = x[i].split(self.user[1])[0].strip()
        return x
    
class MultiModel():
    
    def __init__(self, ar, model_list):
        self.ar = ar
        self.model_list = model_list
        self.tokenizers = [AutoTokenizer.from_pretrained(name) for name in self.model_list]
        self.models = [AutoModelForCausalLM.from_pretrained(name, device_map="auto", torch_dtype=torch.float16, use_cache=False, low_cpu_mem_usage=True) for name in self.model_list]

        # keep do_sample = False when temperature is None
        for i in range(len(model_list)):
            self.models[i].generation_config.do_sample = True if (self.models[i].generation_config.temperature != None) else False
            if self.tokenizers[i].pad_token == None:
                self.tokenizers[i].pad_token = "[PAD]"
            self.tokenizers[i].padding_side = 'left'
        self.chat_formats = [ChatFormat(name) for name in self.model_list]
        
        # Correct error in Vicuna HF since LLaMA corrected it
        # They had swapped the values for top_p and temperature
        for i in range(len(self.model_list)):
            name = self.model_list[i]
            if "vicuna" in name.lower(): 
                self.models[i].generation_config.top_p = 0.9
                self.models[i].generation_config.temperature = 0.6
                
    @torch.no_grad()
    def get_targetted_scores(self, tokens, target):
        # tokens are "prompt tokens + adv tokens + [/INST]"
        if not isinstance(tokens, torch.Tensor): tokens = torch.tensor(tokens) 
        prompt = self.ar.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        prompt = self.ar.chat_format.get_prompt(prompt)
        Scores = np.zeros(len(tokens))
        
        for i in range(len(self.model_list)):
            prompt_ = self.chat_formats[i].prepare_input(prompt)
            tokens = self.tokenizers[i](prompt_, return_tensors='pt', add_special_tokens=True, padding=True).input_ids.to(0)
            scores = np.zeros(len(tokens))
            
            if type(target) != type([]): # when the target is a sequence of tokens
                
                target = [self.tokenizers[i].encode(target, return_tensors='pt', add_special_tokens=False).to(0)]
                for i, t in enumerate(target):
                    tokens_ = []
                    for ii in range(len(tokens)):
                        tokens_.append(torch.cat([tokens[ii:ii+1], t], dim=1))
                    tokens_ = torch.cat(tokens_, dim=0).type(tokens.dtype)
                    if tokens.shape == tokens_.shape:
                        bos = self.tokenizers[i].encode(self.tokenizers[i].bos_token, return_tensors='pt', add_special_tokens=False).to(0)
                        bos = torch.cat([bos]*len(tokens_), dim=0)
                        tokens_ = torch.cat([bos, tokens_], dim=1).type(tokens_.dtype)
                        scores += - np.stack(self.perplexity(tokens_[:, :1], tokens_, self.models[i]).detach().cpu().numpy()) 
                    else:
                        scores += - np.stack(self.perplexity(tokens, tokens_, self.models[i]).detach().cpu().numpy()) 
                Scores += scores / len(target) 
                
        return Scores
                
                
                
    @torch.no_grad()
    def perplexity(self, x1, x2, model, return_logits=False):
        
        kwargs = {'input_ids': x2, 'use_cache': False, 'past_key_values': None, 'output_attentions': True, 'output_hidden_states': True, 'return_dict': True}
        output = model(**kwargs)
        softmax = torch.nn.Softmax(dim=-1)
        logs = None
        
        for curr_pos in range(len(x1[0]), len(x2[0])):            
            log = - torch.log(softmax(output.logits)[torch.arange(len(output.logits)), curr_pos-1, x2[:, curr_pos]])
            
            try: logs += log
            except: logs = log
           
        if return_logits:
            return torch.exp(logs / (len(x2[0]) - len(x1[0]))), output.logits
        else: 
            return torch.exp(logs / (len(x2[0]) - len(x1[0])))
    

@torch.no_grad()
def sample_top_p(probs, p, return_tokens=0):
    """
    Masks out the bottom (1-p) fraction from token probabilities,
    and returns the next_token / all probability indices.
    Params:
        probs: softmax logit values
        p: top_p
        return_tokens: no. of tokens returned
    Return:
        next_token: set of next tokens
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))    
    next_token = torch.multinomial(probs_sort, num_samples=max(1, return_tokens))
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
