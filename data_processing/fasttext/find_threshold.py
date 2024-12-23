import argparse
import json
import numpy as np

parser = argparse.ArgumentParser("Filter an HF dataset and push the result to the hub")
parser.add_argument("--data_path",type=str, help="data path name")
parser.add_argument("--label_name",type=str, help="output name")
parser.add_argument("--percentage",type=float, help="top how many percent", default=0.1)


if __name__ == "__main__":
    args = parser.parse_args()
    scores_and_length = []
    total_length = 0
    print_score = -100
    for i in range(0,2):
        with open(f"{args.data_path}/{str(i).zfill(5)}.jsonl") as f:

            for line in f:
                scores_and_length.append((json.loads(line)["metadata"][f"__label__{args.label_name}"], len(json.loads(line)["text"])))
                total_length += len(json.loads(line)["text"])
        # find the threshold score coresspond to first 10% length data, e.g. we need to sum the length from the highest score until the sum is greater than 10% of the total length
        print(f"file {i} read finished")
        scores_and_length.sort(key=lambda x: x[0], reverse=True)
        current_sum = 0
        for j in range(0, len(scores_and_length)):
            current_sum += scores_and_length[j][1]
            if current_sum > args.percentage * total_length:
                print(f"{args.percentage} length threshold score: {scores_and_length[j][0]}, thre is total # {j} docs, averaged length: {current_sum/(j+1)}")
                print_score = scores_and_length[j][0]
                break
    print(print_score)
            # if sum([x[1] for x in scores_and_length[:i]]) > 0.1 * total_length:
            #     break
            # print(f"min: {min(scores)}, max: {max(scores)}, 10th percentile: {np.percentile(scores, 10)}, mean : {np.mean(scores)}, 50th percentile: {np.percentile(scores, 50)}, 90th percentile: {np.percentile(scores, 90)}")



