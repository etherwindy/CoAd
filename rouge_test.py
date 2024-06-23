import torch
from bitsandbytes import functional as F
import numpy as np
from tqdm import tqdm  # 使用标准的 tqdm
from torch import nn
from transformers import AutoTokenizer, AutoModel,BitsAndBytesConfig
from datasets import load_dataset
from rouge import Rouge
import torch.nn.utils.prune as prune

device = "cuda" if torch.cuda.is_available() else "cpu"


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

def apply_pruning(model, amount=0.5):
    """
    Prune the model and retain the pruned weights.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model


# 设置模型路径
model_path = "/root/pruned_chatglm3-6b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",
    bnb_4bit_compute_dtype=torch.float16
)
#加载模型

model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
#model = AutoModel.from_pretrained(model_path, trust_remote_code=True,quantization_config=bnb_config)
#model = apply_pruning(model, amount=0.3)
#model = model.to(device)

quantization = NF4Quantization()
quantized_model = quantize_model(model, quantization)

torch.cuda.empty_cache()
model = quantized_model.to(device)


tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载 adgen 数据集
dataset = load_dataset("/root/.cache/huggingface/datasets/HasturOfficial___adgen/default/0.0.0/0aa901b33ff15691ac5e94c6893abb1cb2b8b11b")
json_data = dataset['validation'].train_test_split(test_size=0.1, seed=42)['test']
#rouge测试
rouge = Rouge()
preds=[]
labels=[]
for i, item in enumerate(tqdm(json_data)):
    context = item['content']
    target = item["summary"]
    inputs = tokenizer(context, return_tensors='pt').to('cuda')
    target_token = tokenizer(target, return_tensors='pt')
    targetlen = len(target)
    pred = model.generate(**inputs, max_length=targetlen, do_sample=True)
    answer = tokenizer.batch_decode(pred, skip_special_tokens=True)[0]
    index = answer.find(context)
    if index != -1 :
        index = index + len(context)
        answer = answer[index:]

    #print(answer)
        preds += [' '.join(answer)]
        labels += [' '.join(target)]
        scores = rouge.get_scores(hyps=preds, refs=labels, avg=True)

        result = {key: value['f'] * 100 for key, value in scores.items()}

        print(f"Rouge1: {result['rouge-1']:>0.2f} Rouge2: {result['rouge-2']:>0.2f} RougeL: {result['rouge-l']:>0.2f}\n")


