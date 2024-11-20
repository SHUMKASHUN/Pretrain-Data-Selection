import json
from urllib.parse import urlparse
# import ndjson

id2model2loss = {}
# model_list = ["deepseek-llm-7b-base", "llama-7b", "Llama-2-7b-hf", "Mistral-7B-v0.1", "llama-13b", "Llama-2-13b-hf", "llama-30b", "llama-65b"]
model_list = ["llama-7b", "Llama-2-7b-hf", "llama-13b", "Llama-2-13b-hf", "llama-30b", "llama-65b"]


for i in range(0,len(model_list)):
    for j in range(0,16):
        with open(f"./bpc_calculation_results/{model_list[i]}/{j}.json", "r") as f:
            for line in f:
                data = json.loads(line)
                if data["id"] not in id2model2loss:
                    id2model2loss[data["id"]] = {}
                    # raise ValueError("Duplicate ID")
                id2model2loss[data["id"]][model_list[i]] = data["total_loss"]
                

    print(f"{model_list[i]} bpc results load finished")


id2charnum = {}
id2url = {}
for i in range(0,16):

    with open(f"./bpc_calculation_16/{i}.json", "r") as f:
        for line in f:
            data = json.loads(line)
            if data["id"] not in id2charnum:
                id2charnum[data["id"]] = len(data["text"])
            else:
                raise ValueError("Duplicate ID")
            if data["id"] not in id2url:
                id2url[data["id"]] = urlparse(data["url"]).netloc
            else:
                raise ValueError("Duplicate ID")
            
model2benchmark = {
    # "deepseek-llm-7b-base": {
    #                 "arc_e": 75.42,
    #                 "arc_e_norm": 70.75,
    #                 "hellaSwag": 76.1

    #                                 },
                    "llama-7b": {
                    "arc_e":75.38,
                    "arc_e_norm":72.85,
                    "hellaSwag": 76.2
                                    },
                    "Llama-2-7b-hf": {
                    "arc_e":76.30,
                    "arc_e_norm":74.49,
                    "hellaSwag": 76

                                    },
                    # "Mistral-7B-v0.1": {
                    # "arc_e":80.80,
                    # "arc_e_norm":79.50,
                    # "hellaSwag": 81.1

                    #                 },
                    "llama-13b": {
                    "arc_e":77.27,
                    "arc_e_norm":74.62,
                    "hellaSwag": 79.1


                                    },
                    "Llama-2-13b-hf": {
                    "arc_e":79.46,
                    "arc_e_norm":77.40,
                    "hellaSwag": 79.4,
                                    },
                    "llama-30b": {
                    "arc_e":80.47,
                    "arc_e_norm":78.99,
                    "hellaSwag": 82.6


                                    },
                    "llama-65b": {
                    "arc_e":81.19,
                    "arc_e_norm":79.88,
                    "hellaSwag": 84.2

                                    },                                  
}


def correct_order(model2loss, model2benchmark):
    if len(model2loss) < 6:
        return False
    temp_model2benchmark_sort = sorted(model2benchmark.items(), key=lambda x: x[1]["arc_e"], reverse=True)
    # print(temp_model2benchmark_sort)
    temp_model2loss_sort = sorted(model2loss.items(), key=lambda x: x[1], reverse=False)
    # print(temp_model2loss_sort)
    count = 0
    for i in range(0, len(temp_model2benchmark_sort)):
        if temp_model2loss_sort[i][0] != temp_model2benchmark_sort[i][0] :
            count += 1
            
    if count <= 0:
        return True
    return False
    # return True
    # for model in model2loss.keys():
        
def wrong_order(model2loss, model2brnchmark, key):
    # if len(model2loss) < 6:
    #     return False
    temp_model2benchmark_sort = sorted(model2brnchmark.items(), key=lambda x: x[1][key], reverse=True)
    # print(temp_model2benchmark_sort)
    temp_model2loss_sort = sorted(model2loss.items(), key=lambda x: x[1], reverse=False)
    # print(temp_model2loss_sort)

    count = 0
    for i in range(0, len(temp_model2benchmark_sort)):
        if temp_model2loss_sort[i][0] != temp_model2benchmark_sort[i][0] :
            count += 1
    if count > 4 :
        return True  
    else:
        return False


correct_order_id = []
count = 0
for id in id2model2loss.keys():
    if correct_order(id2model2loss[id], model2benchmark):
        correct_order_id.append(id)
    count += 1
    if (count % 100000 == 0):
        print(f"{count} finished")  
print(len(correct_order_id))

wrong_order_id = []
count = 0
for id in id2model2loss.keys():
    if wrong_order(id2model2loss[id], model2benchmark, "arc_e"):
        wrong_order_id.append(id)
    count += 1
    if (count % 100000 == 0):
        print(f"{count} finished")
print(len(wrong_order_id))

all_data = {}
for i in range(0,16):

    with open(f"./bpc_calculation_16/{i}.json", "r") as f:
        for line in f:
            data = json.loads(line)
            if data["id"] not in all_data:
                all_data[data["id"]] = data["text"]
            else:
                raise ValueError("Duplicate ID")


data_positive = []
data_negative = []
for i in range(0,len(correct_order_id)):
    data_positive.append(all_data[correct_order_id[i]])
for i in range(0,len(wrong_order_id)):
    data_negative.append(all_data[wrong_order_id[i]])

print(f"positive: {len(data_positive)}, negative: {len(data_negative)}")

with open("./fasttext_train.txt","w") as f:
    for i in range(0,len(data_positive)):
        f.write("__label__1 " + data_positive[i].replace("\n", " ") + "\n")
    for i in range(0,len(data_negative)):
        f.write("__label__0 " + data_negative[i].replace("\n", " ") + "\n")

import fasttext
model = fasttext.train_supervised(
    input="./fasttext_train.txt",
    epoch=3,
    lr=0.1,
    wordNgrams=2,
)

model.save_model("./saved_fasttext_model.bin")
