<div align="center"><img src="https://github.com/Maxwe11y/gaslightingLLM/blob/master/framework.png" height="300px"/></div>
<h2 align="center">Can Large Language Model be a Gaslighter?</h2>

<div align="center">
    <a>
        <img alt="Python Versions" src="https://img.shields.io/badge/python-%3E%3D3.8-blue">
    </a>
    <a>
        <img alt="PyTorch Versions" src="https://img.shields.io/badge/pytorch-%3E%3D1.10.0-green">
    </a>
</div>

# Can Large Language Model be a Gaslighter?

## Overview
🔥 This is a repository for our paper ([ Can Large Language Model be a Gaslighter?]()).

>Large language models~(LLMs) have gained human trust due to their capabilities and helpfulness. However, this in turn may allow LLMs to affect users' mindsets by manipulating language. It is termed as gaslighting, a psychological effect. 
In this work, we aim to investigate the vulnerability of LLMs under prompt-based and fine-tuning-based gaslighting attacks. Therefore, we propose a two-stage framework DeepCoG designed to: 1) elicit gaslighting plans from LLMs with the proposed DeepGaslighting prompting template, and 2) acquire gaslighting conversations from LLMs through our Chain-of-Gaslighting method. The gaslighting conversation dataset along with a corresponding safe dataset is applied to fine-tuning-based attacks on open-source LLMs and anti-gaslighting safety alignment on these LLMs. Experiments demonstrate that both prompt-based and fine-tuning-based attacks transform three open-source LLMs into gaslighters. In contrast, we advanced three safety alignment strategies to strengthen~(by $12.05\%$) the safety guardrail of LLMs. Our safety alignment strategies have minimal impacts on the utility of LLMs. Empirical studies indicate that an LLM may be a potential gaslighter, even if it passed the harmfulness test on general dangerous queries.
