import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from datasets import Dataset as Dataset_hf

import math
import json
import os 
# os.environ["HF_HOME"] = "/data/vjuicefs_ai_gpt_nlp/72189907/Environment/.cache/huggingface"

class EvalDataset(Dataset):
    def __init__(self, task_name, block_size, stride, tokenizer, file_num=-1, dtype="auto", vocab_size=None):
        # self.args = args
        self.task_name = task_name
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.file_num = file_num
        self.data = None
        self.stride = stride
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            if vocab_size is not None and vocab_size < 65500:
                self._dtype = np.uint16
            else:
                self._dtype = np.int32
        else:
            self._dtype = dtype

        self.ids = []
        self.token_lens = []
        self.char_number_list = []
        if "dolma" in self.task_name:
            self.key_name ="text"
        else:
            self.key_name ="content"
        self._prepare()
        self.prev_end_loc = 0
        self.seq_len = len(self.data)
        self.begin_loc = 0


    def _prepare(self):
        self._curr_idx = 0
        self._arr = []
        if "dolma" in self.task_name:
            self._raw_dataset = load_dataset('allenai/paloma',"dolma-v1_5",split="val" )
        elif "vivo" in self.task_name:
            self._raw_dataset = []

        if (self.task_name == "dolma_cc"):
            self.raw_dataset = self._raw_dataset.filter(lambda example: example['subdomain'] == 'common-crawl')
        elif(self.task_name == "dolma_reddit"):
            self.raw_dataset = self._raw_dataset.filter(lambda example: example['subdomain'] == 'reddit_uniform')
        elif(self.task_name == "dolma_wiki"):
            self.raw_dataset = self._raw_dataset.filter(lambda example: example['subdomain'] == 'wiki')
        elif(self.task_name == "dolma_stack"):
            self.raw_dataset = self._raw_dataset.filter(lambda example: example['subdomain'] == 'stack_uniform')
        elif(self.task_name == "vivo_worldknowledge"):
            with open(f"/data/vjuicefs_ai_gpt_nlp/72189907/Pretrain-Data-Selection/evaluation/sub_dev_data/pure_en_world_knowledge.jsonl", "r") as f:
                    for line in f:
                        data = json.loads(line)
                        if len(data[self.key_name]) > 0:
                            data[self.key_name] = data[self.key_name]
                            self._raw_dataset.append(data)
            self.raw_dataset =  Dataset_hf.from_list(self._raw_dataset)   
        elif(self.task_name == "vivo_code"):
            with open(f"/data/vjuicefs_ai_gpt_nlp/72189907/Pretrain-Data-Selection/evaluation/sub_dev_data/pure_code.jsonl", "r") as f:
                    for line in f:
                        data = json.loads(line)
                        if len(data[self.key_name]) > 0:
                            data[self.key_name] = data[self.key_name]
                            self._raw_dataset.append(data)
            self.raw_dataset =  Dataset_hf.from_list(self._raw_dataset)      
        elif(self.task_name == "vivo_qa"):
            with open(f"/data/vjuicefs_ai_gpt_nlp/72189907/Pretrain-Data-Selection/evaluation/sub_dev_data/pure_en_qa.jsonl", "r") as f:
                    for line in f:
                        data = json.loads(line)
                        if len(data[self.key_name]) > 0:
                            data[self.key_name] = data[self.key_name]
                            self._raw_dataset.append(data)
            self.raw_dataset =  Dataset_hf.from_list(self._raw_dataset)   
        elif(self.task_name == "vivo_news"):
            with open(f"/data/vjuicefs_ai_gpt_nlp/72189907/Pretrain-Data-Selection/evaluation/sub_dev_data/pure_en_news.jsonl", "r") as f:
                    for line in f:
                        data = json.loads(line)
                        if len(data[self.key_name]) > 0:
                            data[self.key_name] = data[self.key_name]
                            self._raw_dataset.append(data)
            self.raw_dataset =  Dataset_hf.from_list(self._raw_dataset)   
        elif(self.task_name == "vivo_novel"):
            with open(f"/data/vjuicefs_ai_gpt_nlp/72189907/Pretrain-Data-Selection/evaluation/sub_dev_data/pure_en_novel.jsonl", "r") as f:
                    for line in f:
                        data = json.loads(line)
                        if len(data[self.key_name]) > 0:
                            data[self.key_name] = data[self.key_name]
                            self._raw_dataset.append(data)
            self.raw_dataset =  Dataset_hf.from_list(self._raw_dataset)   
        elif(self.task_name == "vivo_math"):
            with open(f"/data/vjuicefs_ai_gpt_nlp/72189907/Pretrain-Data-Selection/evaluation/sub_dev_data/pure_en_math.jsonl", "r") as f:
                    for line in f:
                        data = json.loads(line)
                        if len(data[self.key_name]) > 0:
                            data[self.key_name] = data[self.key_name]
                            self._raw_dataset.append(data)
            self.raw_dataset =  Dataset_hf.from_list(self._raw_dataset)   
        elif(self.task_name == "vivo_all"):
            with open(f"/data/vjuicefs_ai_gpt_nlp/72189907/Pretrain-Data-Selection/evaluation/sub_dev_data/pure_all.jsonl", "r") as f:
                    for line in f:
                        data = json.loads(line)
                        if len(data[self.key_name]) > 0:
                            data[self.key_name] = data[self.key_name]
                            self._raw_dataset.append(data)
            self.raw_dataset =  Dataset_hf.from_list(self._raw_dataset)    
        
        self.character_num = 0
        for i in range(len(self.raw_dataset)):
            self.character_num += len(self.raw_dataset[i][self.key_name])

        self.data = self.raw_dataset.map(
            lambda example: {"encoding": np.array(self.tokenizer.encode(example[self.key_name]), dtype=self._dtype)}, num_proc=8)

        self.data = np.concatenate([a['encoding'] for a in self.data], axis=0)

 
    def __len__(self):
        return math.floor((len(self.data)-self.block_size)/self.stride+1)

    def __getitem__(self,item):
        end_loc = min(self.begin_loc+self.block_size, self.seq_len)
        trg_len = end_loc - self.prev_end_loc
        input_ids = self.data[self.begin_loc:end_loc]
        attention_mask = np.ones((len(input_ids),), dtype=bool)
        attention_mask[:-trg_len] = False
        self.prev_end_loc = end_loc
        self.begin_loc = self.begin_loc + self.stride
        return torch.tensor(input_ids), torch.tensor(attention_mask, dtype=bool)
