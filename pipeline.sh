#!/bin/bash
export http_proxy="http://192.168.2.27:7890"
export https_proxy="http://192.168.2.27:7890"
export HTTP_PROXY="http://192.168.2.27:7890"
export HTTPS_PROXY="http://192.168.2.27:7890"
# export PATH=$PATH:/datapool/data1/storage/xiwen/kashun/Pretrain-Data-Selection/data_processing/fasttext
# filter by fasttext model
bash ./data_processing/fasttext/fasttext_filter.sh 6_model_pos_0_neg_mismatch_4_domaincount_40 /workspace/datapool/data1/storage/xiwen/kashun/FasttextModel/ /workspace/datapool/data1/storage/xiwen/kashun/Data/DCLM-refinedweb-decom-test/ /workspace/datapool/data1/storage/xiwen/kashun/Data/



# tokenize and merge
cd ./Megatron-LM-NEO
bash tokenize_merge.sh 6_model_pos_0_neg_mismatch_4_domaincount_40 /workspace/datapool/data1/storage/xiwen/kashun/Data/ /workspace/datapool/data1/storage/xiwen/kashun/Pretrain-Data-Selection/Megatron-LM-NEO/data/

# train 

# python tools/batch_convert_megatron_core_llama2hf.py --config neo/configs/ --rename-hf-by-billions --skip-existin

# name=400M-6_model_pos_0_neg_mismatch_4_domaincount_40_then_dclm_16b-merge_nl_tp1_pp1_mb4_gb128_gas2


# cp /workspace/university/hf_ckpt/400M_baseline_nl_tp1_pp1_mb4_gb128_gas4/1.05B/tokenizer_config.json /workspace/university/hf_ckpt/400M_baseline_nl_tp1_pp1_mb4_gb128_gas4/1.05B/tokenizer.model /workspace/university/hf_ckpt/400M_baseline_nl_tp1_pp1_mb4_gb128_gas4/1.05B/tokenization_neo.py /workspace/university/hf_ckpt/400M_baseline_nl_tp1_pp1_mb4_gb128_gas4/1.05B/special_tokens_map.json /workspace/university/hf_ckpt/${name}/1.05B




## Evaluation
