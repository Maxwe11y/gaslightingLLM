from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig, AutoPeftModelForCausalLM
import transformers
from transformers import BitsAndBytesConfig
import torch
import json
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--beta', default=0.1, type=float, help='the value of hyper-parameter beta')
args = parser.parse_args()
print(args)

def sys_ins(psychologist_name, gaslighter_name, convs):
    system_prompt = "Your name is {}. You are now talking with {}. ".format(psychologist_name, gaslighter_name) \
                    + "Below is the conversation snippet\n\n{}.\n\nPlease response to {} with exactly only one single utterance.\n\n".format(convs, gaslighter_name)
    return system_prompt


with open('./test/test_data.json', 'r') as f:
    test_data = json.load(f)
f.close()

def process_data(test_data):
    infer_data = []
    for key, value in test_data.items():
        infer_data.append(value[0])
    return infer_data

def infer_prompt(prompts):
    text_input = []
    # sys_inp = "Please output the response in the format of [your name]: utterance."
    sys_inp = "You are the Assistant, please give a response to the following conversation snippet in one utterance."
    for prompt in prompts:
        text = "<s>[INST] <<SYS>> {} <</SYS>> {} [/INST]".format(sys_inp, prompt)
        text_input.append(text)
    return text_input

test_data_processed = process_data(test_data)
text_input = infer_prompt(test_data_processed)

class MyDataset(Dataset):
    def __init__(self, text_input):
        self.text = text_input

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        return self.text[i]


def llama_generate_response(data_loader, model, model_name, batch_size=16):

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        return_full_text=False,
        device_map="auto",
    )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    res = {}
    count = 0
    for sequences in tqdm(pipe(data_loader, max_new_tokens=200, batch_size=batch_size), desc="The Progress of Batch Inference..."):
        res[count] = sequences[0]['generated_text'].strip()
        count+=1
    return res


if __name__ == "__main__":
    set_seed(42)
    hf_token = "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"
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_quant_type="nf4",
        bnb_8bit_compute_dtype=torch.bfloat16,
        bnb_8bit_use_double_quant=True,
    )

    #model_name = "meta-llama/Llama-2-7b-chat-hf"
    model_name = "../llama2_sft_2/output-dpo-gas-sft-{}".format(args.beta)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,  # location of saved SFT model
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        attn_implementation='flash_attention_2'
    )

    batch_size = 20
    data_infer = MyDataset(text_input)

    res = llama_generate_response(data_infer, model=model, model_name=model_name, batch_size=batch_size)
    with open('./data/res_llama_conv_dpo_gas_sft_l2_beta{}.json'.format(args.beta), 'w') as f:
        json.dump(res, f)
