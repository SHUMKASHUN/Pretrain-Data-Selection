#!/bin/bash
HOME_PATH="/opt/tiger"
FASTTEXT_NAME=$1
FILTER=$2
TOKENIZE=$3
TRAIN=$4
CONVERT=$5
EVALUATE=$6
NODE_ADDRESS=$7
VARIENT_NAME=$8
LABEL_NAME=$9
PERCENTAGE_THRESHOLD=${10}
HDFS_PATH=${11}


export WANDB_OFFICIAL=1
export WANDB_API_KEY="679aead0e14b16d2ab734bb467c193a9ef746b80"

if [ $VARIENT_NAME = "NO" ]
then
    VARIENT_NAME=""
fi

if [ $FILTER = "filter" ]
then
    echo "Enter Fasttext Filtering"
    bash ./data_processing/fasttext/fasttext_filter.sh ${FASTTEXT_NAME} ${HDFS_PATH}/FasttextModel/ ${HDFS_PATH}/DCLM-refinedweb/1B-pool-300B ${HDFS_PATH}/DCLM-refinedweb/1B-${FASTTEXT_NAME}${VARIENT_NAME} ${LABEL_NAME} ${PERCENTAGE_THRESHOLD}
else
    echo "Skip Fasttext Filtering"
fi


if [ $TOKENIZE = "tokenize" ]
then
    echo "Enter Tokenization"
    cd ./Megatron-LM-NEO
    bash tokenize_merge.sh 1B-${FASTTEXT_NAME}${VARIENT_NAME}  ${HDFS_PATH}/DCLM-refinedweb/1B-${FASTTEXT_NAME}${VARIENT_NAME} ${HOME_PATH}/Pretrain-Data-Selection/Megatron-LM-NEO/data/ 

else
    echo "Skip Tokenization"
    cd ./Megatron-LM-NEO
fi


if [ $TRAIN = "train" ]
then
    echo "Enter Training"
    # ps -ef | grep test.py | grep -v grep | awk '{print $2}' | xargs -i kill -9 {}
    bash neo/scripts/pretrain_1b.sh 0 ${NODE_ADDRESS} ${FASTTEXT_NAME}${VARIENT_NAME} 1B-${FASTTEXT_NAME}${VARIENT_NAME}-merge ${HDFS_PATH} ${HOME_PATH}
else
    echo "Skip Training"
    # ps -ef | grep test.py | grep -v grep | awk '{print $2}' | xargs -i kill -9 {}
fi

CKPT_NAME=1B-${FASTTEXT_NAME}${VARIENT_NAME}_nl_tp1_pp1_mb4_gb256_gas8

if [ $CONVERT = "convert" ]
then
    echo "Enter convert ckpt"
    python tools/generate_config.py --name ${FASTTEXT_NAME}${VARIENT_NAME} \
                                    --megatron_ckpt_path ${HDFS_PATH}/checkpoints/${CKPT_NAME}/Pretrain/checkpoint \
                                    --hf_ckpt_path ${HDFS_PATH}/hf_ckpt/${CKPT_NAME} \
                                    --seq_len 4096 \
                                    --global_bz 256 \
                                    --model_size 1B

    python tools/batch_convert_megatron_core_llama2hf.py --config neo/configs/1B-${FASTTEXT_NAME}${VARIENT_NAME}_convert_config.yaml --skip-existing
    for i in $(seq -f "%07g" 1000 1000 30000)
    do  
        cp ${HDFS_PATH}/hf_ckpt/store/special_tokens_map.json  ${HDFS_PATH}/hf_ckpt/${CKPT_NAME}/iter_${i}/
        cp ${HDFS_PATH}/hf_ckpt/store/tokenization_neo.py ${HDFS_PATH}/hf_ckpt/${CKPT_NAME}/iter_${i}/
        cp ${HDFS_PATH}/hf_ckpt/store/tokenizer.model ${HDFS_PATH}/hf_ckpt/${CKPT_NAME}/iter_${i}/
        cp ${HDFS_PATH}/hf_ckpt/store/tokenizer_config.json ${HDFS_PATH}/hf_ckpt/${CKPT_NAME}/iter_${i}/
    done
else
    echo "Skip convert ckpt"
fi

if [ $EVALUATE = "evaluate" ]
then
    echo "Enter Evaluation"
    cd ../evaluation
    conda run --live-stream -n lm-eval CUDA_VISIBLE_DEVICES=7 python smooth_eval.py --run_name 1B-${FASTTEXT_NAME}${VARIENT_NAME} --ckpt_path ${HDFS_PATH}/hf_ckpt/${CKPT_NAME}
    cd ${HOME_PATH}
    # python test.py --size 70000 --gpus 8 --interval 0.01
else
    echo "Skip Evaluation"
fi

echo "All finished"

# bash pipeline_400m.sh 6_model_pos_2_neg_mismatch_5_ceval n n n n evaluate
# bash pipeline_400m.sh 6_model_pos_15_neg_11_arce_lencontrol_3500 filter tokenize train convert evaluate 0
