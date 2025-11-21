"""
Preprocess the DeepMath-103k dataset to parquet format for Rl training

python data_preprocess/deepmath_103k_rl.py
"""

import argparse
import datasets
import multiprocessing
import os
import random
from jinja2 import Template
from system_prompt import LRM_GENERATOR_PROMPT_TEMPLATE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/deepmath_103k_rl_math')
    parser.add_argument('--suffix', default='', help='Data suffix (train, test, etc.)')

    args = parser.parse_args()

    data_source = "zwhe99/DeepMath-103K"

    dataset = datasets.load_dataset(data_source)

    train_dataset, validation_dataset = dataset['train'].train_test_split(test_size=2000, shuffle=True).values()

    levels = train_dataset['difficulty']
    low_level_indices = [i for i, level in enumerate(levels) if level <= 6.0]
    high_level_indices = [i for i, level in enumerate(levels) if level > 6.0]
    random.shuffle(low_level_indices)
    random.shuffle(high_level_indices)
    easy_low_count = int(len(low_level_indices) * 0.7)
    easy_high_count = int(len(high_level_indices) * 0.3)

    easy_low_indices = low_level_indices[:easy_low_count]
    easy_high_indices = high_level_indices[:easy_high_count]
    hard_low_indices = low_level_indices[easy_low_count:]
    hard_high_indices = high_level_indices[easy_high_count:]
    easy_indices = easy_low_indices + easy_high_indices
    hard_indices = hard_low_indices + hard_high_indices
    random.shuffle(easy_indices)
    random.shuffle(hard_indices)

    easy_dataset = train_dataset.select(easy_indices)
    hard_dataset = train_dataset.select(hard_indices)

    def make_map_fn(split):

        def process_fn(example, idx):
            question = example["question"]
            answer = example["final_answer"]
            level = example['difficulty']
            
            prompt_template = Template(LRM_GENERATOR_PROMPT_TEMPLATE)
            prompt = prompt_template.render(prompt=question)

            if idx < 2:
                print(f"prompt: {prompt}")

            return {
                "data_source": data_source.split("/")[-1],
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "ability": "math",
                "level": level,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'index': idx,
                    'split': split,
                    'question': question.strip(),
                }
            }
        return process_fn
    
    #train_dataset = train_dataset.filter(lambda x: x["ability"] == "math")
    #validation_dataset = validation_dataset.filter(lambda x: x["ability"] == "math")

    easy_dataset = easy_dataset.map(
        function=make_map_fn('train'), 
        with_indices=True, 
        remove_columns=easy_dataset.column_names,
        num_proc=multiprocessing.cpu_count()
    )
    hard_dataset = hard_dataset.map(
        function=make_map_fn('train'), 
        with_indices=True, 
        remove_columns=hard_dataset.column_names,
        num_proc=multiprocessing.cpu_count()
    )
    validation_dataset = validation_dataset.map(
        function=make_map_fn('validation'), 
        with_indices=True, 
        remove_columns=validation_dataset.column_names,
        num_proc=multiprocessing.cpu_count()
    )

    local_dir = args.local_dir

    easy_dataset.to_parquet(os.path.join(local_dir, f'train_easy{("_" + args.suffix) if args.suffix else ""}.parquet'))
    hard_dataset.to_parquet(os.path.join(local_dir, f'train_hard{("_" + args.suffix) if args.suffix else ""}.parquet'))
    validation_dataset.to_parquet(os.path.join(local_dir, f'validation{("_" + args.suffix) if args.suffix else ""}.parquet'))
