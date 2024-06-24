import torch
import time
from transformers import AutoTokenizer, AutoModel
from llmtask import TaskGenerator
import torch.nn.utils.prune as prune
import os



def log(msg):

    with open("mmlu_5shot_pruning_7.log", "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")

def get_model_params_count(model):
    """
    Calculate the total number of parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def get_non_zero_params_count(model):
    """
    Calculate the total number of non-zero parameters in the model.
    """
    non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
    return non_zero_params

def apply_pruning(model, amount=0.7):
    """
    Prune the model and retain the pruned weights.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

device = "cuda"

model = AutoModel.from_pretrained("/root/chatglm3-6b_lora_sft", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/root/chatglm3-6b_lora_sft", trust_remote_code=True)

# Log the number of parameters before pruning
initial_params_count = get_model_params_count(model)
print(initial_params_count)
log(f"Initial model parameters count: {initial_params_count}")

# Apply absolute value based pruning
model = apply_pruning(model, amount=0.3)
model = model.to(device)

# Log the number of parameters after pruning
pruned_params_count = get_model_params_count(model)
non_zero_params_count = get_non_zero_params_count(model)
print(pruned_params_count)
print(non_zero_params_count)
log(f"Pruned model parameters count: {pruned_params_count}")
log(f"Non-zero pruned model parameters count: {non_zero_params_count}")

pruned_model_path = "/root/pruned_chatglm3-6b"
os.makedirs(pruned_model_path, exist_ok=True)
model.save_pretrained(pruned_model_path)
tokenizer.save_pretrained(pruned_model_path)
'''
TG = TaskGenerator("mmlu", max_shot=5)
cnt = 0
for task in TG:
    model_inputs = tokenizer([task], return_tensors="pt").to(device)
    input_tokens = len(model_inputs['input_ids'][0])
    t0 = time.time()
    generated_ids = model.generate(**model_inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    ans = tokenizer.batch_decode([generated_ids[0][input_tokens:]])[0]
    log(f"[{cnt:5}] [{(time.time() - t0):5.3f} s] => ans:{ans}")
    cnt += 1
    TG.feedback(ans)
    log(TG.summary())
    torch.cuda.empty_cache()
    '''