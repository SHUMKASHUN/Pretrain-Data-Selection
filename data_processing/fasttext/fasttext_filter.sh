# example script
# bash fasttext_pipeline.sh pos_exactorder_neg_mismatch_4 /university/Pretrain/datatrove/Fasttext-Model/ /university/DCLM-refinedweb/DCLM-refinedweb-400M-fasttext-pool-80B/ /university/DCLM-refinedweb/
# export HF_ENDPOINT=https://hf-mirror.com

FASTTEXT_MODEL_NAME=$1   ## pos_exactorder_neg_mismatch_4
FASTTEXT_MODEL_PATH=$2  ## /university/Pretrain/datatrove/Fasttext-Model/
POOL_PATH=$3 # /university/DCLM-refinedweb/DCLM-refinedweb-400M-fasttext-pool-80B/
OUTPUT_PATH=$4 # /university/DCLM-refinedweb/${MODEL}
FASTTEXT_LABEL=$5 
PERCENTAGE=$6

if [ ! -f ${FASTTEXT_MODEL_PATH}/${FASTTEXT_MODEL_NAME}.bin ]; then
    echo "Fasttext Model ${FASTTEXT_MODEL_NAME} not found in ${FASTTEXT_MODEL_PATH}, start downloading"
    wget https://huggingface.co/ksshumab/cluster/resolve/main/${FASTTEXT_MODEL_NAME}.bin?download=true -O ${FASTTEXT_MODEL_PATH}${FASTTEXT_MODEL_NAME}.bin
else
    echo "Fasttext Model ${FASTTEXT_MODEL_NAME} already exist in ${FASTTEXT_MODEL_PATH}, skip downloading"
fi



echo "Begin scoring for each docs"
conda run --live-stream -n datatrove python ./data_processing/fasttext/filter.py --input_path ${POOL_PATH}\
    --fasttext  ${FASTTEXT_MODEL_PATH}${FASTTEXT_MODEL_NAME}\
    --output_path ${OUTPUT_PATH} \
    --label_name ${FASTTEXT_LABEL}\
    --threshold -1000

echo "Finish scoring for each docs, finding a threshold correspond to top ${PERCENTAGE} data..."

THRESHOLD=$(python ./data_processing/fasttext/find_threshold.py --data_path  ${OUTPUT_PATH} --label_name ${FASTTEXT_LABEL} --percentage ${PERCENTAGE})  
echo ${THRESHOLD}

THRESHOLD=$(echo ${THRESHOLD} | rev | cut -d' ' -f 1| rev)
echo "Find the new threshold:  ${THRESHOLD}"

conda run --live-stream -n datatrove python ./data_processing/fasttext/filter.py --input_path ${POOL_PATH}\
    --fasttext  ${FASTTEXT_MODEL_PATH}${FASTTEXT_MODEL_NAME}\
    --output_path ${OUTPUT_PATH} \
    --label_name ${FASTTEXT_LABEL} \
    --threshold ${THRESHOLD}

