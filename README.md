## 安装
#### 配置环境
将代码拷贝后，创建虚拟环境：
```bash
conda create -n tango python==3.10
conda activate tango
```
然后在虚拟环境中执行setup.py安装所需要的库：
```bash
cd Efficient_RL_Tango
pip install .
```
之后安装vllm、ninja和flash-attn这三个库：
```bash
pip install -e '.[vllm]'
pip install ninja
pip install flash-attn --no-build-isolation
```

#### 下载模型
下载这几个模型，将${your_base_models_path}替换为您常用的存放模型的文件夹：
```bash
huggingface-cli download DeepSeek-R1-Distill-Qwen-1.5B \
  --local-dir ${your_base_models_path}/DeepSeek-R1-Distill-Qwen-1.5B \
  --local-dir-use-symlinks False

huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir ${your_base_models_path}/Qwen2.5-7B-Instruct \
  --local-dir-use-symlinks False
```

#### 加载数据集
运行这个python文件加载并预处理分别用于rl的数据集：
```bash
# training dataset
python data_preprocess/deepmath_103k_rl.py
```
然后加载用来评估的数据集：
```bash
# evaluation datasets
mkdir -p ./data/StrategyQA
wget -P ./data/StrategyQA https://huggingface.co/datasets/voidful/StrategyQA/resolve/main/strategyqa_train.json
python data_preprocess/prepare_strategyqa.py

mkdir -p ./data/TableBench
wget -P ./data/TableBench https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench/resolve/main/TableBench.jsonl
python data_preprocess/prepare_eval_benchmarks.py
```

## 训练（单节点）
### Stage1
运行
```bash
# similar to above
export VLLM_ATTENTION_BACKEND=XFORMERS
```
之后运行以下命令，需要将.sh中的VERIFIER_MODEL_PATH替换为您的Qwen2.5-7B-Instruct的模型路径，将${your_DeepSeek-R1-Distill-Qwen-1.5B_model_path}替换为您下载的DeepSeek-R1-Distill-Qwen-1.5B的模型路径：
```bash
# on the master node, run
bash scripts/node_1/stage1_run_rl_tango.sh ${your_DeepSeek-R1-Distill-Qwen-1.5B_model_path}
```

### Stage2
运行
```bash
# similar to above
export VLLM_ATTENTION_BACKEND=XFORMERS
```
之后运行以下命令，需要将.sh中的VERIFIER_MODEL_PATH替换为您的Qwen2.5-7B-Instruct的模型路径，将${your_stage1_model_path}替换为您stage1训练好的模型路径（选择./checkpoints/RL-Tango-Stage1/rl-tango-training文件夹中最后一次保存的模型路径）：
```bash
# on the master node, run
bash scripts/node_1/stage2_run_rl_tango.sh ${your_stage1_model_path}
```

## 训练（多节点）
### Stage1
首先在每个节点上运行
```bash
# similar to above
export VLLM_ATTENTION_BACKEND=XFORMERS
```
在主节点上运行
```bash
ray start --head
```
在非主节点上运行
```bash
ray start --address ${MASTER_NODE_ADDRESS}:6379
```
之后在主节点上运行以下命令，需要将.sh中的VERIFIER_MODEL_PATH替换为您的Qwen2.5-7B-Instruct的模型路径，将${your_DeepSeek-R1-Distill-Qwen-1.5B_model_path}替换为您下载的DeepSeek-R1-Distill-Qwen-1.5B的模型路径：
```bash
# on the master node, run
bash scripts/node_4/stage1_run_rl_tango.sh ${your_DeepSeek-R1-Distill-Qwen-1.5B_model_path}
```

### Stage2 – RL训练LLM
还是首先在每个节点上运行
```bash
# similar to above
export VLLM_ATTENTION_BACKEND=XFORMERS
```
在主节点上运行
```bash
ray start --head
```
在非主节点上运行
```bash
ray start --address ${MASTER_NODE_ADDRESS}:6379
```
之后在主节点上运行以下命令，需要将.sh中的VERIFIER_MODEL_PATH替换为您的Qwen2.5-7B-Instruct的模型路径，将${your_stage1_model_path}替换为您stage1训练好的模型路径（选择./checkpoints/RL-Tango-Stage1/rl-tango-training文件夹中最后一次保存的模型路径）：
```bash
# on the master node, run
bash scripts/node_4/stage2_run_rl_tango.sh ${your_stage1_model_path}
```
