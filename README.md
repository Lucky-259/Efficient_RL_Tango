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
huggingface-cli download Qwen/Qwen2.5-Math-1.5B \
--local-dir ${your_base_models_path}/Qwen2.5-Math-1.5B \
--local-dir-use-symlinks False

huggingface-cli download Qwen/Qwen2.5-7B \
  --local-dir ${your_base_models_path}/Qwen2.5-7B \
--local-dir-use-symlinks False

huggingface-cli download meta-llama/Llama-3.1-70B-Instruct \
  --local-dir ${your_base_models_path}/Llama-3.1-70B-Instruct \
--local-dir-use-symlinks False

huggingface-cli download agentica-org/DeepScaleR-1.5B-Preview \
  --local-dir ${your_base_models_path}/DeepScaleR-1.5B-Preview \
--local-dir-use-symlinks False

huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --local-dir ${your_base_models_path}/DeepSeek-R1-Distill-Llama-70B \
--local-dir-use-symlinks False

huggingface-cli download Qwen/Qwen2.5-Math-7B \
--local-dir ${your_base_models_path}/Qwen2.5-Math-7B \
--local-dir-use-symlinks False
```

#### 加载数据集
分别运行以下这两个python文件加载并预处理分别用于sft和rl的数据集：
```bash
# training datasets
python data_preprocess/eurus2_sft.py
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

## 训练
### Stage1 – SFT训练LLM和LRM
#### 产生用于sft LLM的数据
运行以下代码，先产生sft的数据：
先在每个需要运行的机器上运行以下命令：
```bash
# export on all nodes before starting ray
export VLLM_ATTENTION_BACKEND=XFORMERS
```
在主节点上启动ray：
```bash
# launch the master node of ray
ray start --head
```
然后在每个非主节点的机器上运行以下命令启动ray，将${MASTER_NODE_ADDRESS}
换成主节点的ip地址：
```bash
# add the other 3 nodes to the ray cluster
ray start --address ${MASTER_NODE_ADDRESS}:6379
```
之后在主节点上运行（将.sh里的model_path修改为您的Llama-3.1-70B-Instruct
的模型路径）：
```bash
# on the master node, run
bash scripts/qwen_math_1.5b/run_sft_data_generation.sh
```
之后在主节点上运行以下命令来将sft数据分割成train和test：
```bash
# split SFT data into train/test splits
python data_preprocess/split_parquet.py \
    --input ./data/eurus2_sft_math/llama70b_sft_data_generation.parquet
```

#### 产生用于sft LRM的数据
运行以下代码，先产生sft的数据：
先在每个需要运行的机器上运行以下命令：
```bash
# export on all nodes before starting ray
export VLLM_ATTENTION_BACKEND=XFORMERS
```
在主节点上启动ray：
```bash
# launch the master node of ray
ray start --head
```
然后在每个非主节点的机器上运行以下命令启动ray，将${MASTER_NODE_ADDRESS}
换成主节点的ip地址：
```bash
# add the other 3 nodes to the ray cluster
ray start --address ${MASTER_NODE_ADDRESS}:6379
```
之后在主节点上运行（将.sh中的model_path替换为您的DeepSeek-R1-Distill-Llama-70B模型路径）：
```bash
# on the master node, run
bash scripts/deepscaler/run_sft_data_generation.sh
```
之后在主节点上运行以下命令来将sft数据分割成train和test：
```bash
# split SFT data into train/test splits
python data_preprocess/split_parquet.py \
--input ./data/eurus2_sft_math/deepseek_r1_distill_llama70b_sft_data_generation.parquet
```

#### SFT训练 1.5B LLM
在每个节点启动一次，将${i}替换成从0开始的节点序号，将${MASTER_NODE_ADDRESS}替换成主节点的ip地址，将.sh中的model_path替换成您的Qwen2.5-Math-1.5B模型的路径：
```bash
# on node i=0,1,2,3, run
bash scripts/qwen_math_1.5b/run_sft_generator.sh --nnodes 4 \
--node_rank ${i} --master_addr ${MASTER_NODE_ADDRESS}
```

#### SFT训练 7B LLM
在每个节点启动一次，将${i}替换成从0开始的节点序号，将${MASTER_NODE_ADDRESS}替换成主节点的ip地址，将.sh中的model_path替换成您的Qwen2.5-Math-7B模型的路径：
```bash
# on node i=0,1,2,3, run
bash scripts/qwen_math_7b/run_sft_generator.sh --nnodes 4 \
--node_rank ${i} --master_addr ${MASTER_NODE_ADDRESS}
```

#### SFT训练LRM
在每个节点启动一次，将${i}替换成从0开始的节点序号，将${MASTER_NODE_ADDRESS}替换成主节点的ip地址，将.sh中的model_path替换为您的DeepScaleR-1.5B-Preview模型路径：
```bash
# on node i=0,1,2,3, run
bash scripts/deepscaler/run_sft_generator.sh --nnodes 4 --node_rank ${i} \
    --master_addr ${MASTER_NODE_ADDRESS}
```

### Stage2 – RL训练LLM
#### RL训练-1.5B
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
之后在主节点上运行以下命令，需要将.sh中的VERIFIER_MODEL_PATH替换为您的Qwen2.5-7B的模型路径：
```bash
# on the master node, run
bash scripts/qwen_math_1.5b/run_rl_tango.sh ./checkpoints/RL-Tango/sft-generator-qwen-math-1.5b
```

#### RL训练-7B
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
之后在主节点上运行以下命令，需要将.sh中的VERIFIER_MODEL_PATH替换为您的Qwen2.5-7B的模型路径：
```bash
# on the master node, run
bash scripts/qwen_math_7b/run_rl_tango.sh ./checkpoints/RL-Tango/sft-generator-qwen-math-7b
```
