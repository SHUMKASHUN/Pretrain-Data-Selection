import os
# os.environ['HF_DATASETS_OFFLINE ']= "1"
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["HF_HOME"] = "/data/vjuicefs_ai_gpt_nlp/72189907/Environment/.cache/huggingface"
from datasets import Dataset
from datasets import load_dataset, get_dataset_config_names
import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import numpy as np

import wandb
import argparse
from packed_dataset import EvalDataset

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
    elif task == "gsm8k":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = ""
            temp["output"] =  "Question: " + dataset[i]["question"] + "\nAnswer: " + dataset[i]["answer"].split("####")[0].strip("\n") + "\nThe answer is " + dataset[i]["answer"].split("####")[1].strip(" ") + "."
            output_dataset.append(temp) 
    elif task == "math":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = ""
            temp["output"] =  "Question: " + dataset[i]["problem"] + "\nAnswer: " + dataset[i]["solution"]
            output_dataset.append(temp)
    elif task == "bbh":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = ""
            temp["output"] =  "Question: " + dataset[i]["input"] + "\nAnswer: "  + dataset[i]["target"]
            output_dataset.append(temp)        
    elif task == "humaneval":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = ""
            temp["output"] =  "Question: " + dataset[i]["prompt"] + "\nAnswer: " + dataset[i]["canonical_solution"]
            output_dataset.append(temp)   
    elif task == "mbpp":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] =  ""
            temp["output"] ="Question: " + dataset[i]["text"] + "\nAnswer: " +  dataset[i]["code"]
            output_dataset.append(temp) 
    return output_dataset


def preprocess_dataset_answer_only(dataset,task):
    output_dataset = []
    mapping = {"A":0,"B":1,"C":2,"D":3}
    if task == "arc_e":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = "Question: " + dataset[i]["question"] + "\nAnswer: "
            temp["output"] =  dataset[i]["choices"]["text"][ dataset[i]["choices"]["label"].index(dataset[i]["answerKey"])]
            output_dataset.append(temp)
    elif task=="mmlu":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = "Question: " + dataset[i]["question"] + "\nAnswer: "
            temp["output"] =  dataset[i]["choices"][dataset[i]["answer"]]
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
            temp["input"] = "Question: " + dataset[i]["sentence"] + "\nAnswer: "
            if dataset[i]["answer"] == "1":
                temp["output"] =  dataset[i]["option1"]
            elif dataset[i]["answer"] == "2":
                temp["output"] =  dataset[i]["option2"]
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
            temp["input"] = "Context: " + dataset[i]["context"] + "\nQuestion: " + dataset[i]["question"] + "\nAnswer: " 
            temp["output"] =  dataset[i][mapping[dataset[i]["label"]]]
            output_dataset.append(temp)
    elif task == "piqa":
        for i in range(0,len(dataset)):
            mapping = {"0": 'sol1', "1": 'sol2'}
            temp = {}
            temp["input"] =  "Question: " + dataset[i]["goal"] + "\nAnswer: "
            temp["output"] = dataset[i][mapping[str(dataset[i]["label"])]]
            output_dataset.append(temp)
    elif task =="openbookqa":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = "Question: " + dataset[i]["question_stem"] + "\nAnswer: " 
            temp["output"] = dataset[i]["choices"]["text"][ dataset[i]["choices"]["label"].index(dataset[i]["answerKey"])]
            output_dataset.append(temp)
    elif task == "sciq":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = "Question: " + dataset[i]["question"] + "\nAnswer: "
            temp["output"] =  dataset[i]["correct_answer"]
            output_dataset.append(temp)
    elif task == "gsm8k":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = "Question: " + dataset[i]["question"] + "\nAnswer: "
            temp["output"] =  dataset[i]["answer"].split("####")[0].strip("\n") + "\nThe answer is " + dataset[i]["answer"].split("####")[1].strip(" ") + "."
            output_dataset.append(temp) 
    elif task == "math":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = "Question: " + dataset[i]["problem"] + "\nAnswer: "
            temp["output"] =  dataset[i]["solution"]
            output_dataset.append(temp)
    elif task == "bbh":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = "Question: " + dataset[i]["input"] + "\nAnswer: " 
            temp["output"] =  dataset[i]["target"]
            output_dataset.append(temp)        
    elif task == "humaneval":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] = "Question: " + dataset[i]["prompt"] + "\nAnswer: "
            temp["output"] =  dataset[i]["canonical_solution"]
            output_dataset.append(temp)   
    elif task == "mbpp":
        for i in range(0,len(dataset)):
            temp = {}
            temp["input"] =  "Question: " + dataset[i]["text"] + "\nAnswer: "
            temp["output"] = dataset[i]["code"]
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
    task, model_path, mode="answer_only"
):
    model, tokenizer = load_model(model_path)

    if task == "arc_e":
        dataset = load_dataset("allenai/ai2_arc","ARC-Easy", split="test",trust_remote_code=True)
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
    elif task == "gsm8k":
        dataset = load_dataset("openai/gsm8k",'main', split="test",trust_remote_code = True)
    elif task == "math":
        dataset = load_dataset("lighteval/MATH",'all', split="test",trust_remote_code = True)
    elif task == "bbh":
        configs = ['boolean_expressions',
            'causal_judgement',
            'date_understanding',
            'disambiguation_qa',
            'dyck_languages',
            'formal_fallacies',
            'geometric_shapes',
            'hyperbaton',
            'logical_deduction_five_objects',
            'logical_deduction_seven_objects',
            'logical_deduction_three_objects',
            'movie_recommendation',
            'multistep_arithmetic_two',
            'navigate',
            'object_counting',
            'penguins_in_a_table',
            'reasoning_about_colored_objects',
            'ruin_names',
            'salient_translation_error_detection',
            'snarks',
            'sports_understanding',
            'temporal_sequences',
            'tracking_shuffled_objects_five_objects',
            'tracking_shuffled_objects_seven_objects',
            'tracking_shuffled_objects_three_objects',
            'web_of_lies',
            'word_sorting']
        final_dataset = []
        for config in configs:
            dataset = load_dataset("lukaemon/bbh",config,split="test",trust_remote_code = True)
            if mode == "answer_only":
                out = preprocess_dataset_answer_only(dataset,"bbh")
            else:
                out = preprocess_dataset(dataset,"bbh")
            final_dataset.extend(out)
    elif task == "humaneval":
        dataset = load_dataset("openai/openai_humaneval",split="test",trust_remote_code = True)
    elif task == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp","full",split="test",trust_remote_code = True)

    if task == "bbh":
        dataset = Dataset.from_list(final_dataset)
    else:
        if mode == "answer_only":
            dataset = preprocess_dataset_answer_only(dataset,task)
        else:
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


@torch.no_grad()
def calculate_dev_loss(task, model_path):

    model, tokenizer = load_model(model_path)
    valdataset = EvalDataset(
        task_name=task,
        block_size=1900 + 1,
        tokenizer=tokenizer,
        stride=512,
        vocab_size=tokenizer.vocab_size,
    )
    valdataloader = DataLoader(valdataset, batch_size=1, shuffle=False)

    def cross_entropy(logits, targets, attention_mask: torch.Tensor = None):
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(-1)
            targets = targets.masked_fill(~attention_mask, -1)

        return torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1, reduction='sum')

    losses = []
    for k, (val_data, attention_mask) in enumerate(tqdm(valdataloader)):
        input_ids = val_data[:, 0: 1900].contiguous().to("cuda")
        targets = val_data[:, 1: 1900 + 1].contiguous().long().to("cuda")
        attention_mask = attention_mask[:, 1: 1900 + 1].contiguous().to("cuda")
        logits = model(input_ids).logits
        loss = cross_entropy(logits, targets, attention_mask=attention_mask)
        loss = loss.cpu().item()
        losses.append(loss)
        # print("%.8f" % loss)

    out = np.array(losses).sum()
    bpc = out / (valdataset.character_num * np.log(2)) 
    return bpc



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
    tasks = [ "arc_e", "mmlu","lambda", "winogrande", "hellaswag", "siqa", "piqa", "openbookqa", "sciq", "gsm8k", "math", "bbh", "humaneval", "mbpp","dolma_cc", "dolma_reddit", "dolma_wiki", "dolma_stack"]
    #task = 
    # tasks = [ "arc_e", "mmlu","lambda", "winogrande", "hellaswag", "siqa", "piqa", "openbookqa", "sciq", "gsm8k", "math", "bbh", "humaneval", "mbpp","dolma_cc", "dolma_reddit", "dolma_wiki", "dolma_stack","vivo_worldknowledge", "vivo_code", "vivo_qa", "vivo_news", "vivo_novel", "vivo_math"]
    modes = ["with_question", "answer_only"]
    for i in range(0,30):
        # if (i+1) in already:
        #     continue
        model_path=f"{args.ckpt_path}/{id2ckpt[str(i+1)]}" + "/"
        for task in tasks:
            try:
                if "dolma" in task or "vivo" in task:
                    loss = calculate_dev_loss(task, model_path)
                    wandb.log({f"{task}" : round(loss,3), "custom_step": i+1})

                else:
                    for mode in modes:
                        loss = label_model_loss_only_output(task, model_path,mode=mode)
                        wandb.log({f"{task}_{mode}" : round(loss,3), "custom_step": i+1})

            except Exception as e :
                print(f"some error: {e}")
                loss = 0
            # All_loss.append(loss)
            print(f"{args.run_name} {id2ckpt[str(i+1)]} on task {task} has loss {loss}")


