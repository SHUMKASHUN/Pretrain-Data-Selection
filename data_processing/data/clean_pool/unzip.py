# Importing required module
import random
import os
import subprocess
import sys
count = 0
for i in range(1,11):
    for j in range(0,10):
        random.seed(i*10 + j)
        L1 = random.sample(range(1, 3000), 2999)

        for k in range(0,40):
            # /data/vjuicefs_nlp/72189907
            index = str(L1[k]).zfill(8)
            global_index = str(i).zfill(2)
            decompressed_file_name = f"global_{global_index}_local_{j}_shard_{index}_processed.jsonl"
            compressed_file_name = f"global_{global_index}_local_{j}_shard_{index}_processed.jsonl"
            if not os.path.exists("/mnt/hdfs/byte_tiktok_aiic/user/huangyuzhen/data_selection/DCLM-refinedweb/1B-pool-300B/" + decompressed_file_name):
                try:
                    subprocess.Popen(f"cp /mnt/hdfs/byte_tiktok_aiic/user/huangyuzhen/data_selection/DCLM-refinedweb/1B-dclm_original/{compressed_file_name} /mnt/hdfs/byte_tiktok_aiic/user/huangyuzhen/data_selection/DCLM-refinedweb/1B-pool-300B/{decompressed_file_name}",shell=True)
                    print(f"Number {k}: {compressed_file_name}  downloaded")
                except:
                    print(f"{compressed_file_name} download failed")