import torch
from torch import nn
import time
from transformers import AutoTokenizer, AutoModel,BitsAndBytesConfig
from llmtask import TaskGenerator
from tqdm import tqdm
from bitsandbytes import functional as F

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

def log(msg):
    with open("mmlu_5shot_nf4&fp4.log", "a") as f:
        f.write(f"{msg}\n")

model = AutoModel.from_pretrained("/storage/nvme/chatglm_finetune/chatglm3-6b", trust_remote_code=True)
#model = AutoModel.from_pretrained("/root/chatglm3-6b",trust_remote_code=True).quantize(4).cuda()
tokenizer = AutoTokenizer.from_pretrained("/storage/nvme/chatglm_finetune/chatglm3-6b", trust_remote_code=True)

quantization = NF4Quantization()
quantized_model = quantize_model(model, quantization)
torch.cuda.empty_cache()
quantized_model = quantized_model.to(device).eval()

print("nf4_num:", nf4_num)
print("fp4_num:", fp4_num)

#response, history = model.chat(tokenizer, "你好", history=[])
#print(response)
#exit()

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

