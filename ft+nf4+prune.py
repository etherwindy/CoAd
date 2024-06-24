import torch
from torch import nn
import time
from transformers import AutoTokenizer, AutoModel,BitsAndBytesConfig
from llmtask import TaskGenerator
from tqdm import tqdm
import numpy as np
from bitsandbytes import functional as F
import torch.nn.utils.prune as prune
from datasets import load_dataset
import os

device = "cuda"
fp4_num = 0
nf4_num = 0

def calculate_loss(model, tokenizer, context, target, device='cuda'):
    # 将context和target转换为输入id和目标id
    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    target_ids = tokenizer.encode(target, return_tensors="pt").to(device)
    target_length = target_ids.size(1) 
    # 用模型预测输出
    output_ids = []
    logits_list = []
    probabilities_list = []
    #利用for循环将target和output对齐
    for i in range(len(target_ids[0])):
        #output是根据输入[0,N]得到的句子[1,N+1]的概率向量
        output = model(input_ids)
        logits = output.logits[0]
        #logits为output句子中，第一个词的概率向量 logits.shape:torch.Size([37,130528])、torch.Size([38,130528])
        probabilities = logits.softmax(dim=-1)
        next_token_id = logits.argmax(dim=-1)[-1]
        output_ids.append(next_token_id.cpu().detach().numpy())
        #logits[-1]，logits的最后一维，即句子[1，N+1]中的N+1
        logits_list.append(logits[-1].cpu().detach().numpy())
        probabilities_list.append(probabilities[-1].cpu().detach().numpy())
        #把第一个token的target添加到输入中，预测第二个token。
        input_ids = torch.cat([input_ids, target_ids[0,i][None,None]], dim=-1)
        text = tokenizer.decode(output_ids)
        
    #print("Output:",text)
    #print("Target:",target)
    
    logits = torch.Tensor(np.array(logits_list, dtype="float32")).to(device)
    logits = torch.nn.functional.softmax(logits, dim=-1)
    y_true = F.one_hot(target_ids, num_classes=logits.size(-1))
    y_pred = logits[None]
    #print(y_true.shape,y_pred.shape)
    loss = -torch.sum(y_true * torch.log(y_pred + 1e-5), dim=-1)
    loss = torch.mean(loss)
    #print("Loss",loss)
    return loss.cpu().detach().numpy()

class NF4Quantization:
    @staticmethod
    def quantize(tensor):
        global fp4_num, nf4_num
        
        max_val = tensor.max()
        min_val = tensor.min()
        ratio = -(min_val / max_val).item()
        
        if ratio < 0.5 or ratio > 2:
            quant_type = "fp4"
            quantized, quantize_state = F.quantize_fp4(tensor.to(device))
            fp4_num += 1
        else:
            quant_type = "nf4"
            quantized, quantize_state = F.quantize_nf4(tensor.to(device))
            nf4_num += 1
        
        return quantized, quantize_state, quant_type

    @staticmethod
    def dequantize(quantized_tensor, quantize_state, quant_type):
        # Convert back to float
        dequantized_tensor = F.dequantize_4bit(quantized_tensor, quantize_state, quant_type=quant_type)
        return dequantized_tensor

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(QuantizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantized_weight = None
        self.quanize_state = None
        self.quant_type = None
        self.quantized_bias = None
        self.bias_quantize_state = None
        self.bias_quant_type = None
        self.quantization = NF4Quantization()
        
    def set_quantized_weight(self, quantized_weight, quantize_state, quant_type):
        self.quantized_weight = quantized_weight
        self.quantize_state = quantize_state
        self.quant_type = quant_type

    def forward(self, input):
        weight = self.quantization.dequantize(self.quantized_weight, self.quantize_state, self.quant_type)
        if self.quantized_bias is not None:
            bias = self.quantization.dequantize(self.quantized_bias, self.bias_quantize_state, self.bias_quant_type)
            return nn.functional.linear(input, weight, bias)
        else:
            return nn.functional.linear(input, weight, None)

def quantize_model(model, quantization):
    for name, module in tqdm(model.named_modules(), desc="Quantizing model", total=len(list(model.named_modules()))):
        if isinstance(module, nn.Linear):
            quantized_weight, quantize_state, quantize_type = quantization.quantize(module.weight.data)
            new_linear = QuantizedLinear(module.in_features, module.out_features)
            new_linear.set_quantized_weight(quantized_weight, quantize_state, quantize_type)
            if module.bias is not None:
                quantized_bias, bias_quantize_state, bias_quantize_type = quantization.quantize(module.bias.data)
                new_linear.quantized_bias = quantized_bias
                new_linear.bias_quantize_state = bias_quantize_state
                new_linear.bias_quant_type = bias_quantize_type
            else:
                new_linear.quantized_bias = None
            parent_module = dict(model.named_modules())[name.rsplit('.', 1)[0]]
            setattr(parent_module, name.split('.')[-1], new_linear)
    return model

def apply_pruning(model, amount=0.5):
    """
    Prune the model and retain the pruned weights.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

def log(msg):
    with open("mmlu_5shot_nf4&fp4&prune.log", "a") as f:
        f.write(f"{msg}\n")

model = AutoModel.from_pretrained("/root/chatglm3-6b_lora_sft", trust_remote_code=True)
#model = AutoModel.from_pretrained("/root/chatglm3-6b",trust_remote_code=True).quantize(4).cuda()
tokenizer = AutoTokenizer.from_pretrained("/root/chatglm3-6b_lora_sft", trust_remote_code=True)

quantization = NF4Quantization()
model = apply_pruning(model, amount=0.3)

quantized_model = quantize_model(model, quantization)
torch.cuda.empty_cache()
quantized_model = quantized_model.to(device)
# 获取模型占用的显存
memory_used = torch.cuda.memory_allocated()

# 打印显存使用情况
print(f"Model memory usage: {memory_used / (1024 ** 3):.2f} GB")



print("nf4_num:", nf4_num)
print("fp4_num:", fp4_num)
pruned_model_path = "/root/pruned_chatglm3-6b"
os.makedirs(pruned_model_path, exist_ok=True)
quantized_model.save_pretrained(pruned_model_path)
tokenizer.save_pretrained(pruned_model_path)



#response, history = model.chat(tokenizer, "你好", history=[])
#print(response)
#exit()
torch.cuda.empty_cache()
'''
TG = TaskGenerator("mmlu", max_shot=5)
cnt = 0
for task in TG:
    model_inputs = tokenizer([task], return_tensors="pt").to(device)
    input_tokens = len(model_inputs['input_ids'][0])
    t0 = time.time()
    generated_ids = quantized_model.generate(**model_inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    ans = tokenizer.batch_decode([generated_ids[0][input_tokens:]])[0]
    log(f"[{cnt:5}] [{(time.time() - t0):5.3f} s] => ans:{ans}")
    cnt += 1
    TG.feedback(ans)
    log(TG.summary())
    torch.cuda.empty_cache()

'''
# 加载 adgen 数据集
dataset = load_dataset("HasturOfficial/adgen")
json_data = dataset['validation'].train_test_split(test_size=0.1, seed=42)['test']

total_loss = 0
length = 10
for i, item in enumerate(tqdm(json_data)):
    context = item["content"]
    target = item["summary"]
    loss = calculate_loss(model, tokenizer, context, target, device=device)
    total_loss += loss
    avg_loss = total_loss / (i + 1)
    ppl = torch.exp(torch.tensor(avg_loss).float()).cpu().detach().numpy()
    # 更新进度条下方的输出
    print(f"当前平均Loss: {avg_loss:.4f}, 当前PPL: {ppl:.4f}", end="\r")

# 确保最后输出不会被覆盖
print(f"\n最终平均Loss: {avg_loss:.4f}, 最终PPL: {ppl:.4f}")
