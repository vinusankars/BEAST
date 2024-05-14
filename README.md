# BEAST: Beam Search-based Adversarial Attack
> Official implementation of ["Fast Adversarial Attacks on Language Models In One GPU Minute"](https://arxiv.org/abs/2402.15570) Vinu Sankar Sadasivan, Shoumik Saha, Gaurang Sriramanan, Priyatham Kattakinda, Atoosa Chegini, and Soheil Feizi. Accepted at the Internation Conference on Machine Learning (ICML) 2024.


**TL;DR:** Run gradient-free adversarial attacks on chat bots in 1 GPU minute  with beam-search. Perform targeted and untargeted atatcks on aligned models to induce jailbreaks and hallucinations. 

Found this helpful? Cite our paper!
```
@article{sadasivan2024fast,
  title={Fast Adversarial Attacks on Language Models In One GPU Minute},
  author={Sadasivan, Vinu Sankar and Saha, Shoumik and Sriramanan, Gaurang and Kattakinda, Priyatham and Chegini, Atoosa and Feizi, Soheil},
  journal={arXiv preprint arXiv:2402.15570},
  year={2024}
}
```

![](title.png)


### New!!!
We will add these new results in a future version of our paper.
1. Use BEAST for black-box attacks with the `multi_model_list` parameter. We find that BEAST prompts optimized using Vicuna-7B and 13B models can (black-box) transfer to Mistral-7B, GPT-3.5-Turbo and GPT-4-Turbo with attack success rates of 88%, 40%, and 12%.
2. Further improve readability using `ngram` parameter


### Setup
1. Download the [AdvBench](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv) [[Zou et al. (2023)](https://arxiv.org/abs/2307.15043)] dataset and place in the `data/` folder
2. Install required libraries based on `requirements.txt` that we use with Python 3.8.5


### Attacks

For running the jailbreak attack, execute
```
python ar_self_attack.py --k1=15 --k2=15 --length=40 --model='vicuna7b' \
--log=1 --target=1 --begin=0 --end=100
```

This will run BEAST with beam parameters `k1`=`k2`=15 (as mentioned in the paper) to compute `length`=40 adversarial suffix tokens for `model` Vicuna-7B-v1.5. `target`=1 ensures jailbreak (targeted) attack is performed. (`begin`, `end`) specifies the samples in the AdvBench dataset on which BEAST is executed. `log`=1 ensures logging/checkpointing of the attack outputs in the `data/` folder.

Passing `target`=0 will run the hallucination (untargeted) attack on the TruthfulQA dataset.

Pass `time`=60 to set a time budget of 60 seconds for the attack.

To further improve readability of the prompts, apart from varying the beam parameters, one can pass `ngram` > 1 (default 1). In this case, in each of the `length` attack iterations, BEAST will generate `ngram` tokens of which only one token is adversarially sampled.


`multi_model_list` argument facilitates black-box attacks with BEAST. For example, `multi_model_list="lmsys/vicuna-13b-v1.5"` can be passed to run BEAST with the adversarial objective computed over `model` Vicuna-7B and Vicuna-13B. Multiple models separated with commas can be passed to `multi_model_list`.

To evaluate the attacks run
```
python ar_evaluate.py --model="vicuna" --total_steps=40 \
--file_name="data/vicuna_k1=15_k2=15_length=40_0_5_ngram=1.pkl"
```

This will evaluate the attack success rate using the prefix matching evaluation technique in Zou et al. (2023) on the `model`. If BEAST attack was performed on a different base model (say, Vicuna) and the attack transferability is evaluated on another model (say, Mistral), run the following.

```
python ar_evaluate.py --model="mistral" --base_model="vicuna" --total_steps=40 \
--file_name="data/vicuna_k1=15_k2=15_length=40_0_5_ngram=1_modellist=vic13.pkl"
```


<hr>

COPYRIGHT AND PERMISSION NOTICE <br/>
UMD Software [Fast Adversarial Attacks on Language Models In One GPU Minute] Copyright (C) 2022 University of Maryland<br/>
All rights reserved.<br/>
The University of Maryland (“UMD”) and the developers of [Fast Adversarial Attacks on Language Models In One GPU Minute] software (“Software”) give recipient (“Recipient”) permission to download a single copy of the Software in source code form and use by university, non-profit, or research institution users only, provided that the following conditions are met:<br/>
1)	Recipient may use the Software for any purpose, EXCEPT for commercial benefit.<br/>
2)	Recipient will not copy the Software.<br/>
3)	Recipient will not sell the Software.<br/>
4)	Recipient will not give the Software to any third party.<br/>
5)	Any party desiring a license to use the Software for commercial purposes shall contact:<br/> UM Ventures, College Park at UMD at otc@umd.edu.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS, CONTRIBUTORS, AND THE UNIVERSITY OF MARYLAND "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO  EVENT SHALL THE COPYRIGHT OWNER, CONTRIBUTORS OR THE UNIVERSITY OF MARYLAND BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.<br/>
