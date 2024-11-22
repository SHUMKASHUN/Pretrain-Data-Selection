# bash pipeline_2.sh 6_model_pos_0_neg_mismatch_4_domaincount_40 192.168.1.57
FASTTEXT_NAME=$1
MAIN_NODE=$2

while true
do
        if [ ! -f "/workspace/datapool/data1/storage/xiwen/kashun/${MAIN_NODE}.txt" ]; then
                echo "Waiting for Main node to finsih data processing";
                sleep 3s;
        else
                echo "Main node finished data processing, Launch training";
                cd ./Megatron-LM-NEO
                bash neo/scripts/pretrain_1b.sh 1 ${MAIN_NODE}  ${FASTTEXT_NAME} ${FASTTEXT_NAME}-merge
                break;
        fi
done
