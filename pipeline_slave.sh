# bash pipeline_2.sh 6_model_pos_0_neg_mismatch_4_domaincount_40 192.168.1.57
HOME_PATH="/opt/tiger"
export FASTTEXT_NAME=$1
FILTER=$2
TOKENIZE=$3
TRAIN=$4
CONVERT=$5
EVALUATE=$6
NODE_ADDRESS=$7
export VARIENT_NAME=$8
LABEL_NAME=$9
PERCENTAGE_THRESHOLD=${10}
export HDFS_PATH=${11}
N_NODE=${12}
TRAINING_STEPS=${13}
NODE_RANK=${14}

if [ $VARIENT_NAME = "NO" ]
then
    VARIENT_NAME=""
fi

while true
do
        if [ ! -f "${HDFS_PATH}/${ARNOLD_WORKER_0_HOST}_${FASTTEXT_NAME}${VARIENT_NAME}.txt" ]; then
                echo "Waiting for Main node to finsih data processing";
                sleep 3s;
        else
                echo "Main node finished data processing, Launch training";
                cd ./Megatron-LM-NEO
                
                mkdir -p ${HOME_PATH}/Pretrain-Data-Selection/Megatron-LM-NEO/data/1B-${FASTTEXT_NAME}${VARIENT_NAME}-merge
                hdfs dfs -get hdfs://harunasg/home/byte_tiktok_aiic/user/huangyuzhen/data_selection/data/1B-${FASTTEXT_NAME}${VARIENT_NAME}-merge  ${HOME_PATH}/Pretrain-Data-Selection/Megatron-LM-NEO/data/
                # cp -r /mnt/hdfs/byte_tiktok_aiic/user/huangyuzhen/data_selection/data/1B-${FASTTEXT_NAME}${VARIENT_NAME}-merge ${HOME_PATH}/Pretrain-Data-Selection/Megatron-LM-NEO/data/
                bash neo/scripts/pretrain_1b_multi.sh ${N_NODE} ${NODE_RANK} ${FASTTEXT_NAME}${VARIENT_NAME} 1B-${FASTTEXT_NAME}${VARIENT_NAME}-merge ${HDFS_PATH} ${HOME_PATH} ${TRAINING_STEPS}
                break;
        fi
done
