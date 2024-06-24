import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn.utils.prune as prune
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader

def log(msg):
    with open("adgen_sensitive_pruning.log", "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")

def get_model_params_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def get_non_zero_params_count(model):
    non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
    return non_zero_params

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(dataloader)

def forward_hook(module, input, output):
    module.output = output

def register_hooks(model):
    hooks = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            hook = module.register_forward_hook(forward_hook)
            hooks.append(hook)
    return hooks

def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()

def print_memory_usage():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")


def sensitivity_analysis(model, dataloader, device):
    sensitivities = {}
    model.train()  # 需要计算梯度，因此将模型设置为训练模式
    hooks = register_hooks(model)
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            sensitivities[name] = 0.0

    for batch in tqdm(dataloader, desc="Sensitivity Analysis"):

        inputs = tokenizer(batch['content'], return_tensors='pt', padding=True, truncation=True).to(device)
        labels = tokenizer(batch['summary'], return_tensors='pt', padding=True, truncation=True).to(device)['input_ids']

        # Adjusting labels to match the input size
        labels = labels[:, :inputs['input_ids'].shape[1]]

        # Ensure inputs and labels are on the same device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        # 前向传播计算所有层的输出
        outputs = model(**inputs, labels=labels)
        
        loss = outputs.loss  # 计算损失

        # 逐层计算并记录梯度，然后删除该层的梯度
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                module.zero_grad()  # 清除现有的梯度
                if hasattr(module, 'output'):
                    module_output = module.output
                else:
                    continue

                if module_output.grad_fn is not None:
                    grad_output = torch.ones_like(module_output).to(device)
                    torch.autograd.backward([module_output], [grad_output], retain_graph=True)

                    if module.weight.grad is not None:
                        sensitivities[name] += module.weight.grad.abs().mean().item()

                    # 打印内存使用情况
                    print(f"After backward for {name}:")
                    print_memory_usage()

                    # 删除当前层的梯度以释放内存
                    module.weight.grad = None
                    module_output.grad = None  # 清除输出的梯度
                    del module_output  # 删除输出变量

                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    # 再次打印内存使用情况，确保内存已释放
                    print(f"After clearing gradients for {name}:")
                    print_memory_usage()

        # 清理内存
        model.zero_grad()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    remove_hooks(hooks)

    # 对每一层的敏感度进行平均处理
    total_batches = len(dataloader)
    for name in sensitivities:
        sensitivities[name] /= total_batches

    return sensitivities


def apply_pruning(model, sensitivities, pruning_threshold):
    for name, module in model.named_modules():
        if name in sensitivities:
            # 计算层的敏感度
            module_sensitivity = sensitivities[name]
            if module_sensitivity < pruning_threshold:
                prune.l1_unstructured(module, name='weight', amount=0.5)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 设置模型路径
model = AutoModel.from_pretrained("/root/chatglm3-6b", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/root/chatglm3-6b", trust_remote_code=True)

# Log the number of parameters before pruning
initial_params_count = get_model_params_count(model)
print(f"Initial model parameters count: {initial_params_count}")
log(f"Initial model parameters count: {initial_params_count}")

# 加载 adgen 数据集
dataset = load_dataset("HasturOfficial/adgen")
split_dataset = dataset['train'].train_test_split(test_size=0.0001, seed=42)['test']

# 创建 DataLoader 并设置批次大小
batch_size = 2  # 减小批次大小以减少内存使用
dataloader = DataLoader(split_dataset, batch_size=batch_size)

# 敏感度分析
sensitivities = sensitivity_analysis(model, dataloader, device)

# 应用敏感度剪枝
pruning_threshold = 0.1  # 设定一个剪枝阈值，这里需要根据实际情况调整
apply_pruning(model, sensitivities, pruning_threshold)

# Log the number of parameters after pruning
pruned_params_count = get_model_params_count(model)
non_zero_params_count = get_non_zero_params_count(model)
print(f"Pruned model parameters count: {pruned_params_count}")
print(f"Non-zero pruned model parameters count: {non_zero_params_count}")
log(f"Pruned model parameters count: {pruned_params_count}")
log(f"Non-zero pruned model parameters count: {non_zero_params_count}")

'''
# 在 /root 目录下新建文件夹并保存剪枝后的模型
pruned_model_path = "/root/pruned_chatglm3-6b"
os.makedirs(pruned_model_path, exist_ok=True)
model.save_pretrained(pruned_model_path)
tokenizer.save_pretrained(pruned_model_path)

'''