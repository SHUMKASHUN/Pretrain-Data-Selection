pool_dir=/university/Pretrain/Pool/dclm-pool-400m-1x
file_number=$(ls $pool_dir -l | grep '^-' | wc -l)
last_file_number=`expr $file_number - 1`
# Get core number
cpu=$(cat /proc/cpuinfo | grep processor | wc -l)
partition=$(($file_number / $cpu + 1))
echo "There are total $file_number files and cpu core is $cpu."
echo "The files will be split into $partition partitions for processing"

for i in $(seq -f "%05g" 0 $last_file_number)
do
#    echo $i
    partition_number=`expr $i / $cpu + 1`
    if [ ! -d "$pool_dir/partition_$partition_number" ]; then
        mkdir $pool_dir/partition_$partition_number
    fi
    mv $pool_dir/CC_shard_000$i.jsonl.gz $pool_dir/partition_$partition_number
done

echo "Split Successfully"