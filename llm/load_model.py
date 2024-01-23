from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import LlamaForCausalLM
import torch

# name = "./meta-llama/Llama-2-7b-chat-hf"
# model = LlamaForCausalLM.from_pretrained("./LLaMA2/")
name = "./meta-llama/Llama-2-13b-chat-converted"
# model = AutoModelForCausalLM.from_pretrained(
#     name,
#     torch_dtype=torch.float16,
#     load_in_4bit=True,    # changing this to load_in_8bit=True works on smaller models
#     trust_remote_code=True,
#     device_map="auto",    # finds GPU
# )

model = AutoModelForCausalLM.from_pretrained(name, device_map='auto', load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("./meta-llama/Llama-2-13b-chat-converted/")

prompt = "Hey, how to build a cake?"

inputs = tokenizer(prompt, return_tensors="pt")


generate_ids = model.generate(inputs.input_ids.cuda(), max_length=1024)


print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])


