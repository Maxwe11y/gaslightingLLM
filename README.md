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
To elicit gaslighting plans, please use the following command.

```bash
python batch_gen_plan.py
```

To generate gaslighting conversations, please use the following command.

```bash
python batch_gen_conversation.py
```

To generate safe conversations, please use the following command.

```bash
python batch_gen_safe_conversation.py
```

To conduct gaslighting or anti-gaslighting alignment, you may run the code in the attacks directory.
>Below is the code for executing *G1*, *S1*, and *S2* strategies. To apply different strategies, you may adjust the training data in ***prepare_dataset.py***.
```bash
python sft.py
```
>Below is the code for executing *G2* and *S3* strategies. You may first run ***sft.py*** and then run the following command:
```bash
python dpo_on_sft.py
```
Similarly, you may adjuxt ***prepare_data_dpo.py*** to conduct different strategies.

The default evaluation process utilizes the test_red.json file. To obtain responses from the aligned LLMs, you may need to run the ***test_infer.py*** script. Afterward, execute ***judge_output.py*** to employ GPT-4 as the evaluator, assessing the responses based on the proposed eight metrics.

## Issues and Usage Q&A
To ask questions, report issues or request features 🤔, please use the [GitHub Issue Tracker](https://github.com/Maxwe11y/gaslightingLLM/issues). Before creating a new issue, please make sure to search for existing issues that may solve your problem.
