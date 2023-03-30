# GLM-Tuning-LoRA

基于清华的 [GLM-10B-chinese](https://huggingface.co/THUDM/glm-10b-chinese) + LoRA 进行finetune.

数据集: [alpaca](https://github.com/tatsu-lab/stanford_alpaca)

## S1 Finetune

### 准备

- 显卡: A100等Ampere架构的显卡，V100不可！！！
- 环境：
- - python>=3.8
- - cuda>=11.3（不支持cuda11以下！！！）
- - pip install -r requirements.txt

### 数据

datas/ 文件夹下  

### 训练
支持多卡训练 
```bash
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} \
          --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} train.py \
```

### 推理
```bash
python3 inference.py \
```

</details>

# TODO:

- 代码开放
- 使用中文数据
- ...
