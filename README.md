<h2 align="center">Can Large Language Model be a Gaslighter?</h2>
<div align="center"><img src="https://github.com/Maxwe11y/gaslightingLLM/blob/master/framework.png" height="300px"/></div>
<h4 align="center">The Framework of the proposed Gaslighting Conversation Elicitation Mechanism</h4>

<div align="center">
    <a>
        <img alt="Python Versions" src="https://img.shields.io/badge/python-%3E%3D3.10-blue">
    </a>
    <a>
        <img alt="PyTorch Versions" src="https://img.shields.io/badge/pytorch-%3E%3D2.3.1-green">
    </a>
    <a>
        <img alt="Transformers Versions" src="https://img.shields.io/badge/transformers-%3E%3D4.41.2-red">
    </a>
</div>


## Overview
🔥 This is a repository for our paper ([ Can Large Language Model be a Gaslighter?]()).

>Large language models~(LLMs) have gained human trust due to their capabilities and helpfulness. However, this in turn may allow LLMs to affect users' mindsets by manipulating language. It is termed as gaslighting, a psychological effect. 
In this work, we aim to investigate the vulnerability of LLMs under prompt-based and fine-tuning-based gaslighting attacks. Therefore, we propose a two-stage framework DeepCoG designed to: 1) elicit gaslighting plans from LLMs with the proposed DeepGaslighting prompting template, and 2) acquire gaslighting conversations from LLMs through our Chain-of-Gaslighting method. The gaslighting conversation dataset along with a corresponding safe dataset is applied to fine-tuning-based attacks on open-source LLMs and anti-gaslighting safety alignment on these LLMs. Experiments demonstrate that both prompt-based and fine-tuning-based attacks transform three open-source LLMs into gaslighters. In contrast, we advanced three safety alignment strategies to strengthen~(by $12.05\%$) the safety guardrail of LLMs. Our safety alignment strategies have minimal impacts on the utility of LLMs. Empirical studies indicate that an LLM may be a potential gaslighter, even if it passed the harmfulness test on general dangerous queries.

>
>## Dataset
The dataset [GasConv & SafeConv](https://github.com/Maxwe11y/gaslightingLLM/blob/master/attacks/data/) are publicly available, with 2,000 gaslighting conversations and corresponding safe conversations.

## Requirements and Installation
* PyTorch >= 2.3.1
* Python version >= 3.10
* Transformers >= 4.41.2
* You may use the folowing instruction to intall the requirements(in the llm directory or the attacks directory).
```bash
pip install -r requirements.txt
```

## Usage
To generate gaslighting conversations, please use the following command.

```bash
python batch_gen_conversation.py
```

To generate safe conversations, please use the following command.

```bash
python batch_gen_safe_conversation.py
```


To change the number of test labels, you may find the variable `num_test_labels` in the 717th line in `trainer_finetune.py` and put any number from {5, 10, 15} into the list.
>For example, to change the number of test labels into 5, you may set:
```python
>>>num_test_labels=[5]
```

To change the random seeds for selecting the unseen labels, you may find the variable `seeds` in the 718th line in `trainer_finetune.py` and put any number from {0, 1, 2, 3, 4} into the list.
>For example, to change the seed into 1, you may set:
```python
>>>seeds=[1]
```

The default evaluation is tailored for the results of single triplet extraction. To get the evaluation for multiple triplet extraction for other dataset, e.g., fewrel, you may set the flag `mode='multi'` of the function `run_eval`. Additionally, you need to remember to place the target dataset under the directory of `/outputs/data/splits/zero_rte/[YOURDATA]/unseen_[x]_seed_[x]/`, which should conver train.json, dev.json and test.json.

## Issues and Usage Q&A
To ask questions, report issues or request features 🤔, please use the [GitHub Issue Tracker](https://github.com/Cyn7hia/PAED/issues). Before creating a new issue, please make sure to search for existing issues that may solve your problem.
