import os
# os.environ['HF_DATASETS_OFFLINE ']= "1"
# os.environ["HF_HUB_OFFLINE"] = "1"

from datasets import Dataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import wandb
import argparse
os.environ["WANDB_API_KEY"] = "679aead0e14b16d2ab734bb467c193a9ef746b80"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
parser = argparse.ArgumentParser("Smooth Eval")
parser.add_argument("--run_name",type=str, help="wandb run name")
parser.add_argument("--ckpt_path",type=str, help="ckpt path name")



BATCH_SIZE = 16
def preprocess_dataset(dataset,task):
    output_dataset = []
    mapping = {"A":0,"B":1,"C":2,"D":3}
    if task == "arc_e":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = ""
            temp["output"] = "Question: " + dataset[i]["question"] + "\nAnswer: " + dataset[i]["choices"]["text"][ dataset[i]["choices"]["label"].index(dataset[i]["answerKey"])]
            output_dataset.append(temp)
    elif task=="mmlu":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = ""
            temp["output"] = "Question: " + dataset[i]["question"] + "\nAnswer: " + dataset[i]["choices"][dataset[i]["answer"]]
            output_dataset.append(temp)
    elif task=="lambda":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = ""
            temp["output"] = dataset[i]["text"]
            output_dataset.append(temp)
    elif task=="winogrande":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = ""
            if dataset[i]["answer"] == "1":
                temp["output"] = "Question: " + dataset[i]["sentence"] + "\nAnswer: " + dataset[i]["option1"]
            elif dataset[i]["answer"] == "2":
                temp["output"] = "Question: " + dataset[i]["sentence"] + "\nAnswer: " + dataset[i]["option2"]
            output_dataset.append(temp)
    elif task=="hellaswag":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = ""
            temp["output"] = dataset[i]["ctx"] + " " + dataset[i]["endings"][int(dataset[i]["label"])]
            output_dataset.append(temp)
    elif task =="siqa":
        for i in range(0,len(dataset)):
            mapping = {"1": 'answerA', "2": 'answerB', "3": 'answerC'}
            temp = {}
            temp["input"] = ""
            temp["output"] = "Context: " + dataset[i]["context"] + "\nQuestion: " + dataset[i]["question"] + "\nAnswer: " + dataset[i][mapping[dataset[i]["label"]]]
            output_dataset.append(temp)
    elif task == "piqa":
        for i in range(0,len(dataset)):
            mapping = {"0": 'sol1', "1": 'sol2'}
            temp = {}
            temp["input"] = ""
            temp["output"] = "Question: " + dataset[i]["goal"] + "\nAnswer: " + dataset[i][mapping[str(dataset[i]["label"])]]
            output_dataset.append(temp)
    elif task =="openbookqa":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = ""
            temp["output"] = "Question: " + dataset[i]["question_stem"] + "\nAnswer: " + dataset[i]["choices"]["text"][ dataset[i]["choices"]["label"].index(dataset[i]["answerKey"])]
            output_dataset.append(temp)
    elif task == "sciq":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = ""
            temp["output"] = "Question: " + dataset[i]["question"] + "\nAnswer: " + dataset[i]["correct_answer"]
            output_dataset.append(temp)
    return output_dataset


def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.to("cuda")
    model.eval()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=2048,
        trust_remote_code = True
    )
    return model, tokenizer


def label_model_loss_only_output(
    task, model_path
):
    model, tokenizer = load_model(model_path)

    if task == "arc_e":
        dataset = load_dataset("allenai/ai2_arc","ARC-Easy", split="test",cache_dir="~/.cache/huggingface/datasets",trust_remote_code=True)
    elif task == "mmlu":
        dataset = load_dataset("cais/mmlu","all", split="test",trust_remote_code=True)
    elif task == "lambda":
        dataset = load_dataset("EleutherAI/lambada_openai","en", split="test",trust_remote_code=True)
    elif task == "winogrande":
        dataset = load_dataset("allenai/winogrande", "winogrande_s", split="validation",trust_remote_code=True)
    elif task == "hellaswag":
        dataset = load_dataset("Rowan/hellaswag", split="validation",trust_remote_code=True)
    elif task == "siqa":
        dataset = load_dataset("allenai/social_i_qa", split="validation",trust_remote_code=True)
    elif task == "piqa":
        dataset = load_dataset("ybisk/piqa", split="validation",trust_remote_code=True)
    elif task == "openbookqa":
        dataset = load_dataset("allenai/openbookqa", split="test",trust_remote_code=True)
    elif task == "sciq":
        dataset = load_dataset("allenai/sciq", split="test",trust_remote_code=True)
    dataset = preprocess_dataset(dataset,task)
    dataset = Dataset.from_list(dataset)
    # dataset = load_dataset("json", data_files="/ssddata/ksshumab/Pretrain/arc_easy_demo.jsonl", split="train")
    # preprocess and tokenize the dataset
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    loss_value = []
    with torch.no_grad():
        # batched dataset
        for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
            batch = dataset[i : i + BATCH_SIZE]
            real_BATCH_SIZE = len(batch["input"])
            # tokenize the batch
            batch_inputs = []
            batch_outputs = []
            for j in range(real_BATCH_SIZE):
                inputs_j = tokenizer.encode(
                    batch["input"][j], max_length=2048, add_special_tokens=False, return_tensors="pt"
                ).view(-1)
                outputs_j = tokenizer.encode(
                    batch["output"][j], max_length=2048, add_special_tokens=False, return_tensors="pt"
                ).view(-1)
                batch_inputs.append(inputs_j)
                batch_outputs.append(outputs_j)
            # padding the batch
            max_len_input = max([len(inputs) for inputs in batch_inputs])
            max_len_output = max([len(outputs) for outputs in batch_outputs])
            max_len = max_len_input + max_len_output
            input_ids = torch.zeros(
                real_BATCH_SIZE, max_len, dtype=torch.long, device="cuda"
            )
            length_mask = torch.zeros(
                real_BATCH_SIZE, max_len, dtype=torch.float, device="cuda"
            )
            output_mask = torch.zeros(
                real_BATCH_SIZE, max_len, dtype=torch.float, device="cuda"
            )
            for j in range(real_BATCH_SIZE):
                input_len = len(batch_inputs[j])
                output_len = len(batch_outputs[j])
                input_ids[j, :input_len] = batch_inputs[j]
                input_ids[j, input_len : input_len + output_len] = batch_outputs[j]
                length_mask[j, : input_len + output_len] = 1
                output_mask[j, input_len : input_len + output_len] = 1
            # forward
            output = model(input_ids)
            # calculate loss
            shift_logits = output.logits[..., :-1, :].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = input_ids[..., 1:].contiguous().view(-1)
            loss = loss_fn(shift_logits, shift_labels)
            # reshape as [batch size x seq length]
            loss = loss.view(real_BATCH_SIZE, -1)
            loss = loss * length_mask[..., :-1] * output_mask[..., 1:]
            # average over the sequence length
            loss_list = []
            for k in range(real_BATCH_SIZE):
                output_length = output_mask[k].sum()
                loss_single = loss[k].sum() / output_length
                loss_list.append(loss_single.item())
            loss_value.extend(loss_list)
    return sum(loss_value) / len(loss_value)

if __name__ == "__main__":
    args = parser.parse_args()


    # ALL_NAME = "400M-6_model_pos_0_neg_mismatch_4_domaincount_40_then_dclm_16b-merge_nl_tp1_pp1_mb4_gb128_gas2"
    wandb.init(project="Pretrain-Eval-smooth",name=f"{args.run_name}")

    All_loss = []
    id2ckpt = {"1": "iter_0001000", "2": "iter_0002000", "3": "iter_0003000", "4": "iter_0004000", 
    "5": "iter_0005000", "6": "iter_0006000", "7": "iter_0007000", "8":"iter_0008000", "9":"iter_0009000",
    "10":"iter_0010000", "11":"iter_0011000", "12":"iter_0012000", "13":"iter_0013000", "14":"iter_0014000",
    "15":"iter_0015000", "16":"iter_0016000", "17":"iter_0017000", "18":"iter_0018000", "19":"iter_0019000",
    "20":"iter_0020000", "21":"iter_0021000", "22":"iter_0022000", "23":"iter_0023000", "24":"iter_0024000",
    "25":"iter_0025000", "26":"iter_0026000", "27":"iter_0027000", "28":"iter_0028000", "29":"iter_0029000",
    "30":"iter_0030000", "31":"iter_0031000", "32":"iter_0032000", "33":"iter_0033000", "34":"iter_0034000"}

    tasks = [ "arc_e", "mmlu","lambda", "winogrande", "hellaswag", "siqa", "piqa", "openbookqa", "sciq"]
    # tasks = [ "openbookqa"]

    for i in range(0,30):
        # if (i+1) in already:
        #     continue
        model_path=f"{args.ckpt_path}/{id2ckpt[str(i+1)]}" + "/"
        for task in tasks:
    # file_path = "arc_easy_demo.jsonl"
    #     task = "arc_e"
            try:
                loss = label_model_loss_only_output(task, model_path)
                wandb.log({f"{task}" : round(loss,3), "custom_step": i+1})

            except Exception as e :
                print(f"some error: {e}")
                loss = 0
            # All_loss.append(loss)
            print(f"{args.run_name} {id2ckpt[str(i+1)]} on task {task} has loss {loss}")




