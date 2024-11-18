data_dir=/workspace/university/Pretrain/Pool/dclm-pool-400m-1x
#convert from zstd to gzip, which can be further processed by spark
#Note: 400M pool has disk space ~ 600G for zstd file
for file in $data_dir/* ; do
    file=${file%*/}
    file=${file##*/}
    #echo "$file"
    zstd -d $data_dir/$file
    jsonl_file=${file/".zst"/""}
    #echo $jsonl_file
    gzip $data_dir/$jsonl_file
    rm $data_dir/$file
    echo "$file convert done"
done
