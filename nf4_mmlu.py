import torch
import time
from transformers import AutoTokenizer, AutoModel,BitsAndBytesConfig
from llmtask import TaskGenerator

torch.cuda.set_device(3)

def log(msg):
    with open("mmlu_5shot_nf4.log", "a") as f:
        f.write(f"{msg}\n")

device = "cuda:3"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModel.from_pretrained("/storage/nvme/chatglm_finetune/chatglm3-6b", trust_remote_code=True, quantization_config=bnb_config)

#model = AutoModel.from_pretrained("/root/chatglm3-6b",trust_remote_code=True).quantize(4).cuda()

tokenizer = AutoTokenizer.from_pretrained("/storage/nvme/chatglm_finetune/chatglm3-6b", trust_remote_code=True)

model = model.eval()
#response, history = model.chat(tokenizer, "你好", history=[])
#print(response)

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

