#!/bin/bash
export http_proxy="http://192.168.2.27:7890"
export https_proxy="http://192.168.2.27:7890"
export HTTP_PROXY="http://192.168.2.27:7890"
export HTTPS_PROXY="http://192.168.2.27:7890"
# export PATH=$PATH:/datapool/data1/storage/xiwen/kashun/Pretrain-Data-Selection/data_processing/fasttext
# filter by fasttext model
# bash ./data_processing/fasttext/fasttext_filter.sh 6_model_pos_0_neg_mismatch_4_domaincount_40 /workspace/datapool/data1/storage/xiwen/kashun/FasttextModel/ /workspace/datapool/data1/storage/xiwen/kashun/Data/DCLM-refinedweb-decom-test/ /workspace/datapool/data1/storage/xiwen/kashun/Data/



# tokenize and merge
cd ./Megatron-LM-NEO
# bash tokenize_merge.sh 6_model_pos_0_neg_mismatch_4_domaincount_40 /workspace/datapool/data1/storage/xiwen/kashun/Data/ /workspace/datapool/data1/storage/xiwen/kashun/Pretrain-Data-Selection/Megatron-LM-NEO/data/

# train 
# bash neo/scripts/pretrain_400m.sh 0 172.17.0.1 6_model_pos_0_neg_mismatch_4_domaincount_40_EXP1 6_model_pos_0_neg_mismatch_4_domaincount_40-merge


# convert checkpoint
CKPT_NAME=400M-6_model_pos_0_neg_mismatch_4_domaincount_40_EXP1_nl_tp1_pp1_mb8_gb256_gas4

# python tools/generate_config.py --name 6_model_pos_0_neg_mismatch_4_domaincount_40 \
#                                 --megatron_ckpt_path /workspace/datapool/data1/storage/xiwen/kashun/checkpoints/${CKPT_NAME}/Pretrain/checkpoint \
#                                 --hf_ckpt_path /workspace/datapool/data1/storage/xiwen/kashun/hf_ckpt/${CKPT_NAME} \
#                                 --seq_len 4096 \
#                                 --global_bz 256

# python tools/batch_convert_megatron_core_llama2hf.py --config neo/configs/6_model_pos_0_neg_mismatch_4_domaincount_40_convert_config.yaml --skip-existing

for i in $(seq -f "%07g" 100 100 1000)
do  
    cp ../../hf_ckpt/store/special_tokens_map.json  /workspace/datapool/data1/storage/xiwen/kashun/hf_ckpt/${CKPT_NAME}/iter_${i}/
    cp ../../hf_ckpt/store/tokenization_neo.py /workspace/datapool/data1/storage/xiwen/kashun/hf_ckpt/${CKPT_NAME}/iter_${i}/
    cp ../../hf_ckpt/store/tokenizer.model /workspace/datapool/data1/storage/xiwen/kashun/hf_ckpt/${CKPT_NAME}/iter_${i}/
    cp ../../hf_ckpt/store/tokenizer_config.json /workspace/datapool/data1/storage/xiwen/kashun/hf_ckpt/${CKPT_NAME}/iter_${i}/
done


# name=400M-6_model_pos_0_neg_mismatch_4_domaincount_40_then_dclm_16b-merge_nl_tp1_pp1_mb4_gb128_gas2

# cp /workspace/university/hf_ckpt/400M_baseline_nl_tp1_pp1_mb4_gb128_gas4/1.05B/tokenizer_config.json /workspace/university/hf_ckpt/400M_baseline_nl_tp1_pp1_mb4_gb128_gas4/1.05B/tokenizer.model /workspace/university/hf_ckpt/400M_baseline_nl_tp1_pp1_mb4_gb128_gas4/1.05B/tokenization_neo.py /workspace/university/hf_ckpt/400M_baseline_nl_tp1_pp1_mb4_gb128_gas4/1.05B/special_tokens_map.json /workspace/university/hf_ckpt/${name}/1.05B




## Evaluation
