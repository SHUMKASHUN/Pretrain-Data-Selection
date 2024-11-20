# example script
# bash fasttext_pipeline.sh pos_exactorder_neg_mismatch_4 /university/Pretrain/datatrove/Fasttext-Model/ /university/DCLM-refinedweb/DCLM-refinedweb-400M-fasttext-pool-80B/ /university/DCLM-refinedweb/
export HF_ENDPOINT=https://hf-mirror.com

FASTTEXT_MODEL_NAME=$1   ## pos_exactorder_neg_mismatch_4
FASTTEXT_MODEL_PATH=$2  ## /university/Pretrain/datatrove/Fasttext-Model/
POOL_PATH=$3 # /university/DCLM-refinedweb/DCLM-refinedweb-400M-fasttext-pool-80B/
OUTPUT_PATH=$4 # /university/DCLM-refinedweb/${MODEL}
if [ ! -f ${FASTTEXT_MODEL_PATH}/${FASTTEXT_MODEL_NAME}.bin ]; then
    echo "Fasttext Model ${FASTTEXT_MODEL_NAME} not found in ${FASTTEXT_MODEL_PATH}, start downloading"
    wget https://huggingface.co/ksshumab/cluster/resolve/main/${FASTTEXT_MODEL_NAME}.bin?download=true -O ${FASTTEXT_MODEL_PATH}${FASTTEXT_MODEL_NAME}.bin
else
    echo "Fasttext Model ${FASTTEXT_MODEL_NAME} already exist in ${FASTTEXT_MODEL_PATH}, skip downloading"
fi


# python filter.py --fasttext  ${FASTTEXT_MODEL_PATH}${MODEL} --output_name  /university/DCLM-refinedweb/${MODEL} 
#python filter.py --fasttext  ${MODEL} --output_name  dclm_original --threshold 0.026

echo "Begin scoring for each docs"
conda run --live-stream -n datatrove python filter.py --input_path ${POOL_PATH}\
    --fasttext  ${FASTTEXT_MODEL_PATH}${FASTTEXT_MODEL_NAME}\
    --output_path ${OUTPUT_PATH}${FASTTEXT_MODEL_NAME}\
    --label_name "1"\
    --threshold -1000

echo "Finish scoring for each docs, finding a threshold correspond to top 10% data..."

THRESHOLD=$(python find_threshold.py --data_path  ${OUTPUT_PATH}${FASTTEXT_MODEL_NAME} --label_name "1")
echo ${THRESHOLD}

THRESHOLD=$(echo ${THRESHOLD} | rev | cut -d' ' -f 1| rev)
echo "Find the new threshold:  ${THRESHOLD}"

conda run --live-stream -n datatrove python filter.py --input_path ${POOL_PATH}\
    --fasttext  ${FASTTEXT_MODEL_PATH}${FASTTEXT_MODEL_NAME}\
    --output_path ${OUTPUT_PATH}${FASTTEXT_MODEL_NAME}\
    --label_name "1"\
    --threshold ${THRESHOLD}

