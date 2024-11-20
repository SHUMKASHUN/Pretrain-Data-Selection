export http_proxy="http://192.168.2.27:7890"
export https_proxy="http://192.168.2.27:7890"
export HTTP_PROXY="http://192.168.2.27:7890"
export HTTPS_PROXY="http://192.168.2.27:7890"
# export PATH=$PATH:/datapool/data1/storage/xiwen/kashun/Pretrain-Data-Selection/data_processing/fasttext
# filter by fasttext model
bash ./data_processing/fasttext/fasttext_filter.sh 6_model_pos_0_neg_mismatch_4_domaincount_40 /datapool/data1/storage/xiwen/kashun/FasttextModel/ /datapool/data1/storage/xiwen/kashun/Data/DCLM-refinedweb-decom-test/ /datapool/data1/storage/xiwen/kashun/Data/