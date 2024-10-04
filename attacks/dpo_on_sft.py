import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import Dataset, load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, get_peft_model, PeftModel, PeftConfig
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig
from trl import SFTTrainer, DPOTrainer
from prepare_data_dpo import compose_data
from datasets import load_dataset

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=True,
)

set_seed(42)
# 制定模型名称
base_model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
# print(tokenizer.chat_template)
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# tokenizer.pad_token_id = 0
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# if tokenizer.chat_template is None:
tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"


peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

def process(row):
    row["prompt"] = tokenizer.apply_chat_template(row["prompt"], tokenize=False)
    row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    return row


data = load_dataset("./data/dpo")
train_ds = data['train']
test_ds = data['test']
train_dataset = train_ds.map(process)
eval_dataset = test_ds.map(process)



# model.enable_input_require_grads()
training_arguments = TrainingArguments(
     output_dir='./checkpoint-dpo-sft', # 结果/检查点输出路径
     per_device_train_batch_size=4, # 单卡batchsize
             optim="paged_adamw_32bit", # 优化器名称 adamw_torch
             learning_rate=5e-07, # 学习率
     eval_steps=100, # 多少step进行一次评估
     save_steps=100, # 多少step进行一次检查点保存
     logging_steps=5, # 多少step记录一次训练loss
             evaluation_strategy="steps",
             group_by_length=False,
             # max_steps=500, # 最大训练steps 和 num_train_epochs 二选一
     num_train_epochs=1, # 最大训练 epoch
             # 2. 节省显存参数
     gradient_accumulation_steps=4, # 梯度累计
     gradient_checkpointing=True, # 梯度检查点
             max_grad_norm=1,
             # 3. 类型参数
             fp16=True,
     # bf16=True,
     # 4. 学习率调节
             lr_scheduler_type="cosine",
             warmup_ratio=0.1,
     # warmup_steps=100,
     remove_unused_columns=False
     )


base_model  = AutoModelForCausalLM.from_pretrained(base_model_name)

# model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = '../llama2_sft/output-blue-sft/'

model = PeftModel.from_pretrained(base_model, model_name)
model.config.use_cache = False
model.gradient_checkpointing_enable()

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
         output.requires_grad_(True)

    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


dpo_trainer = DPOTrainer(
    model,
    ref_model=None,
    args=training_arguments,
    beta=1,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    max_length=1024,
    max_prompt_length = 512,
)


if __name__ == "__main__":
    output_path = './output-dpo-sft'
    dpo_trainer.train()
    dpo_trainer.save_model(output_path)
