# Importing required module
import random
import os
import subprocess
import sys
count = 0
for i in range(3,11):
    for j in range(0,10):
        random.seed(i*10 + j)
        L1 = random.sample(range(1, 3000), 2999)

        for k in range(0,30):
            # /data/vjuicefs_nlp/72189907
            index = str(L1[k]).zfill(8)
            global_index = str(i).zfill(2)
            decompressed_file_name = f"global_{global_index}_local_{j}_shard_{index}_processed.jsonl"
            compressed_file_name = f"global_{global_index}_local_{j}_shard_{index}_processed.jsonl.zstd"
            if not os.path.exists("/data/vjuicefs_nlp/72189907/DCLM-refinedweb-decom/" + decompressed_file_name):
                try:
                    subprocess.Popen(f"zstd -d /data/vjuicefs_nlp/72189907/DCLM-refinedweb/{compressed_file_name} -o /data/vjuicefs_nlp/72189907/DCLM-refinedweb-decom/{decompressed_file_name}",shell=True)
                    print(f"Number {k}: {compressed_file_name}  downloaded")
                except:
                    print(f"{compressed_file_name} download failed")




