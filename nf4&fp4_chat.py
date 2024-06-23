import torch
from torch import nn
import time
from transformers import AutoTokenizer, AutoModel,BitsAndBytesConfig
from tqdm import tqdm
from bitsandbytes import functional as F
from datasets import load_dataset
import numpy as np

device = "cuda:3"
fp4_num = 0
nf4_num = 0

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
model = AutoModel.from_pretrained("/storage/nvme/chatglm_finetune/models/chatglm3-6b_lora_sft", trust_remote_code=True)
#model = AutoModel.from_pretrained("/root/chatglm3-6b",trust_remote_code=True).quantize(4).cuda()
tokenizer = AutoTokenizer.from_pretrained("/storage/nvme/chatglm_finetune/models/chatglm3-6b_lora_sft", trust_remote_code=True)

quantization = NF4Quantization()
quantized_model = quantize_model(model, quantization)
torch.cuda.empty_cache()
quantized_model = quantized_model.to(device).eval()

print("nf4_num:", nf4_num)
print("fp4_num:", fp4_num)

# 加载 adgen 数据集
#dataset = load_dataset("HasturOfficial/adgen")
#json_data = dataset['validation'].train_test_split(test_size=0.1, seed=42)['test']
#
#total_loss = 0
#length = 10
#for i, item in enumerate(tqdm(json_data)):
#    context = item["content"]
#    input_tensor = tokenizer(context, return_tensors="pt").to(device)
#    with torch.no_grad():
#        #生成文本
#        generated_ids = quantized_model.generate(**input_tensor, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)

# chat

while True:
    context = input("Usr: ")
    if context == "exit":
        break
    input_tensor = tokenizer(context, return_tensors="pt").to(device)
    reponse, _ = quantized_model.chat(tokenizer, context, history=[])
    print("Bot: ", reponse)
    
# 确保最后输出不会被覆盖
print("Max gpu memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")