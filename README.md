# CoAd

## 模型微调

利用 [Adgen](https://hf-mirror.com/datasets/HasturOfficial/adgen) 数据集，使用 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 中的 QLoRA 对 [chatglm3-6b](https://hf-mirror.com/THUDM/chatglm3-6b)模型进行微调。微调模型所使用的配置文件在`llama-factory`文件夹中。

## 推理

使用自定义量化方法量化微调后的模型并进行推理“

```python
python nf4\&fp4_chat.py
```

使用前需要讲代码中的模型路径替换为微调后的模型路径，并且更改推理所用的 gpu。

## 测试

原模型使用 NF4 量化的 mmlu：

```python
python nf4_mmlu.py
```

原模型使用自定义量化的 mmlu：

```python
python nf4&fp4.mmlu.py
```

微调后的模型使用自定义量化的 ppl:

```python
python nf4&fp4_ppl.py
```