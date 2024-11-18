import asyncio
import sys
import random
import os
import subprocess

MAX_PROCESSES = 5


async def process_download(global_index, local_index, index,file_name,sem):
    async with sem:  # controls/allows running 10 concurrent subprocesses at a time
        try:
            proc = await asyncio.create_subprocess_exec('/home/university/pkgs/bin/aws', 's3', 'cp', f's3://commoncrawl/contrib/datacomp/DCLM-refinedweb/global-shard_{global_index}_of_10/local-shard_{local_index}_of_10/shard_{index}_processed.jsonl.zstd', f'/university/DCLM-refinedweb/train-fasttext-pool/{file_name}')
            await proc.wait()
        except:
            print(f"{file_name} download failed")

async def main():
    count = 0
    sem = asyncio.Semaphore(MAX_PROCESSES)

    for i in range(1,11):
        tasks_download = []
        for j in range(0,10):
            random.seed(i*10 + j)
            L1 = random.sample(range(1, 3000), 2999)
            for k in range(100,110):
                index = str(L1[k]).zfill(8)
                global_index = str(i).zfill(2)
                file_name = f"global_{global_index}_local_{j}_shard_{index}_processed.jsonl.zstd"
                if not os.path.exists("/university/DCLM-refinedweb/train-fasttext-pool/" + file_name):
                    #count += 1
                    tasks_download.append(process_download(global_index, j, index,file_name, sem))

    #print(count)
        await asyncio.gather(*tasks_download)


asyncio.run(main())





