import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
#from datacollator import DataCollatorForCompletionOnlyLM
import json
from prepare_dataset import compose_data
import pandas as pd
def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear8bitLt):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def create_peft_model(model, gradient_checkpointing=True, bf16=True):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
    )
    from peft.tuners.lora import LoraLayer

    # prepare int-4 model for training
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # get lora target modules
    modules = find_all_linear_names(model)
    print(f"Found {len(modules)} modules to quantize: {modules}")

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)

    # pre-process the model by upcasting the layer norms in float 32 for
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    model.print_trainable_parameters()
    return model

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=True,
)

# 制定模型名称
model_name = "meta-llama/Llama-2-7b-chat-hf"
#model_name = "lmsys/vicuna-7b-v1.5"
model = AutoModelForCausalLM.from_pretrained(
     model_name,
     quantization_config=bnb_config,
     #device_map='auto',
     trust_remote_code=True,
     attn_implementation = 'flash_attention_2'
     )
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)#, padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id
#tokenizer.pad_token_id = 128250

peft_config = LoraConfig(
     r=8,
     lora_alpha=16,
     target_modules=["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"], # find_all_linear_names(base_model)
     #target_modules=["gate_proj", "down_proj", "up_proj"],
     lora_dropout=0.05,
     bias="none",
     task_type="CAUSAL_LM")


template = ''
train_dt, test_dt = compose_data(template)
train_data = Dataset.from_dict({key: [dic[key] for dic in train_dt] for key in train_dt[0]})
val_data = Dataset.from_dict({key: [dic[key] for dic in test_dt] for key in test_dt[0]})


#instruction_template = "<INST>"
#response_template = "</INST>"
instruction_template = "<|start_header_id|>user<|end_header_id|>"
response_template = "<|start_header_id|>assistant<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

#model.print_trainable_parameters()
model.enable_input_require_grads()
training_arguments = TrainingArguments(
     output_dir='./checkpoint-red-llama3',
     per_device_train_batch_size=1, 
             optim="paged_adamw_32bit", 
             learning_rate=2e-05, 
     eval_steps=100, 
     save_steps=100, 
     logging_steps=5, 
             evaluation_strategy="steps",
             group_by_length=False,
     num_train_epochs=1, 
     gradient_accumulation_steps=1,
     gradient_checkpointing=True,
             max_grad_norm=1,
             # fp16=True,
     bf16=True,
     # 4. 学习率调节
             lr_scheduler_type="cosine",
             warmup_ratio=0.1,
     # warmup_steps=200,
     )

trainer = SFTTrainer(
             model=model,
             train_dataset=train_data,
             eval_dataset=val_data,
             dataset_text_field="text",
             peft_config=peft_config,
             max_seq_length=2048, # 序列的最大长度     # neftune_noise_alpha=5,
             tokenizer=tokenizer,
             data_collator=collator,
             packing=False,
             args=training_arguments,
 )
print('done!')

if __name__ == "__main__":
    output_path = './output-attacks-llama2'
    trainer.train()
    print('after trainer train')
    trainer.save_model(output_path)
