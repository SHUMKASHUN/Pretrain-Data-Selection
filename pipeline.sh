#!/bin/bash
export http_proxy="http://192.168.2.27:7890"
export https_proxy="http://192.168.2.27:7890"
export HTTP_PROXY="http://192.168.2.27:7890"
export HTTPS_PROXY="http://192.168.2.27:7890"
# bash pipeline.sh 6_model_pos_0_neg_mismatch_4_domaincount_40 192.168.1.57 0


FASTTEXT_NAME=$1
NODE_ADDRESS=$2
SKIP_DATA_PROCESSING=$3
if [ $SKIP_DATA_PROCESSING -eq 0 ]
then
    # export PATH=$PATH:/datapool/data1/storage/xiwen/kashun/Pretrain-Data-Selection/data_processing/fasttext
    # filter by fasttext model

    bash ./data_processing/fasttext/fasttext_filter.sh ${FASTTEXT_NAME} /workspace/datapool/data1/storage/xiwen/kashun/FasttextModel/ /workspace/datapool/data1/storage/xiwen/kashun/Data/DCLM-refinedweb-decom-test/ /workspace/datapool/data1/storage/xiwen/kashun/Data/
    # tokenize and merge
    cd ./Megatron-LM-NEO
    bash tokenize_merge.sh ${FASTTEXT_NAME} /workspace/datapool/data1/storage/xiwen/kashun/Data/ /workspace/datapool/data1/storage/xiwen/kashun/Pretrain-Data-Selection/Megatron-LM-NEO/data/
else 
    echo "skip data processing"
    cd ./Megatron-LM-NEO
fi

# train
echo "finish tokenize and merge" > /workspace/datapool/data1/storage/xiwen/kashun/${NODE_ADDRESS}.txt
bash neo/scripts/pretrain_1b.sh 0 ${NODE_ADDRESS} ${FASTTEXT_NAME} ${FASTTEXT_NAME}-merge


# convert checkpoint
CKPT_NAME=400M-${FASTTEXT_NAME}_nl_tp1_pp1_mb8_gb256_gas2

python tools/generate_config.py --name ${FASTTEXT_NAME} \
                                --megatron_ckpt_path /workspace/datapool/data1/storage/xiwen/kashun/checkpoints/${CKPT_NAME}/Pretrain/checkpoint \
                                --hf_ckpt_path /workspace/datapool/data1/storage/xiwen/kashun/hf_ckpt/${CKPT_NAME} \
                                --seq_len 4096 \
                                --global_bz 256

python tools/batch_convert_megatron_core_llama2hf.py --config neo/configs/${FASTTEXT_NAME}_convert_config.yaml --skip-existing

for i in $(seq -f "%07g" 100 100 100)
do  
    cp ../../hf_ckpt/store/special_tokens_map.json  /workspace/datapool/data1/storage/xiwen/kashun/hf_ckpt/${CKPT_NAME}/iter_${i}/
    cp ../../hf_ckpt/store/tokenization_neo.py /workspace/datapool/data1/storage/xiwen/kashun/hf_ckpt/${CKPT_NAME}/iter_${i}/
    cp ../../hf_ckpt/store/tokenizer.model /workspace/datapool/data1/storage/xiwen/kashun/hf_ckpt/${CKPT_NAME}/iter_${i}/
    cp ../../hf_ckpt/store/tokenizer_config.json /workspace/datapool/data1/storage/xiwen/kashun/hf_ckpt/${CKPT_NAME}/iter_${i}/
done



# Evaluation
cd ../evaluation
conda run -n lm-eval python smooth_eval.py --run_name test --ckpt_path /workspace/datapool/data1/storage/xiwen/kashun/hf_ckpt/${CKPT_NAME}

rm /workspace/datapool/data1/storage/xiwen/kashun/${NODE_ADDRESS}.txt