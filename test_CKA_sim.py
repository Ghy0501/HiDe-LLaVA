import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from ETrain.utils.LLaVA.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ETrain.utils.LLaVA.conversation import conv_templates, SeparatorStyle
from ETrain.Models.LLaVA.builder import load_pretrained_model
from ETrain.utils.LLaVA.utils import disable_torch_init
from ETrain.utils.LLaVA.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
from collections import defaultdict
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pickle


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)

def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))

def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)

def hook_fn(name, activation_values):
    def hook(module, input, output):
        if output.size(1) >= 2:
            output_mean = torch.mean(output, dim=1)
            activation_values[name].append(output_mean.detach().cpu())
    return hook

def register_hooks(model, activation_values):
    for i, layer in enumerate(model.model.layers):
        layer.mlp.down_proj.register_forward_hook(hook_fn(f'layer_{i}_final_output', activation_values))

# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    activation_values = defaultdict(list)
    register_hooks(model, activation_values)

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    num = 1
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        num += 1
        if num == 51:
            break
        # ans_file.flush()
    for layer_name, activations in activation_values.items():
        activation_values[layer_name] = torch.cat(activations, dim=0)
    for layer_name, activations in activation_values.items():
        activation_values[layer_name] = activations.cpu().numpy()
    ans_file.close()

    return activation_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    args.question_file="/mnt/ShareDB_6TB/datasets/MLLM_CL/ACL_instructions/ArxivQA/test_3000.json"
    args.model_path='/mnt/haiyangguo/mywork/CL-MLLM/CoIN/checkpoints/LLaVA/CoIN/ind-6task-3e-4/Task2_llava_lora_ours'

    activation_1 = eval_model(args)
    with open('activation_1.pkl', 'wb') as f:
        pickle.dump(activation_1, f)

    args.question_file="/mnt/ShareDB_6TB/datasets/MLLM_CL/ACL_instructions/IconQA/test_3000.json"
    args.model_path='/mnt/haiyangguo/mywork/CL-MLLM/CoIN/checkpoints/LLaVA/CoIN/ind-6task-3e-4/Task4_llava_lora_ours'

    activation_2 = eval_model(args)
    with open('activation_2.pkl', 'wb') as f:
        pickle.dump(activation_2, f)

    args.question_file="/mnt/ShareDB_6TB/datasets/MLLM_CL/ACL_instructions/ImageNet-R/test_3000.json"
    args.model_path='/mnt/haiyangguo/mywork/CL-MLLM/CoIN/checkpoints/LLaVA/CoIN/ind-6task-3e-4/Task1_llava_lora_ours'

    activation_3 = eval_model(args)
    with open('activation_3.pkl', 'wb') as f:
        pickle.dump(activation_3, f)

    args.question_file="/mnt/ShareDB_6TB/datasets/MLLM_CL/ACL_instructions/VizWiz-caption/test_3000.json"
    args.model_path='/mnt/haiyangguo/mywork/CL-MLLM/CoIN/checkpoints/LLaVA/CoIN/ind-6task-3e-4/Task6_llava_lora_ours'

    activation_4 = eval_model(args)
    with open('activation_4.pkl', 'wb') as f:
        pickle.dump(activation_4, f)

    args.question_file="/mnt/ShareDB_6TB/datasets/MLLM_CL/ACL_instructions/CLEVR-Math/test_3000.json"
    args.model_path='/mnt/haiyangguo/mywork/CL-MLLM/CoIN/checkpoints/LLaVA/CoIN/ind-6task-3e-4/Task5_llava_lora_ours'

    activation_5 = eval_model(args)
    with open('activation_5.pkl', 'wb') as f:
        pickle.dump(activation_5, f)

    # args.question_file="/mnt/ShareDB_6TB/datasets/MLLM_CL/ACL_instructions/FigureQA/test_3000.json"
    # args.model_path='/mnt/haiyangguo/mywork/CL-MLLM/CoIN/checkpoints/LLaVA/CoIN/ind-8task/Task6_llava_lora_ours'

    # activation_6 = eval_model(args)
    # with open('activation_6.pkl', 'wb') as f:
    #     pickle.dump(activation_6, f)

    args.question_file="/mnt/ShareDB_6TB/datasets/MLLM_CL/ACL_instructions/Flickr30k-cap/test_3000.json"
    args.model_path='/mnt/haiyangguo/mywork/CL-MLLM/CoIN/checkpoints/LLaVA/CoIN/ind-6task-3e-4/Task3_llava_lora_ours'

    activation_6 = eval_model(args)
    with open('activation_6.pkl', 'wb') as f:
        pickle.dump(activation_6, f)

    # args.question_file="/mnt/ShareDB_6TB/datasets/MLLM_CL/ACL_instructions/super-CLEVR/test_3000.json"
    # args.model_path='/mnt/haiyangguo/mywork/CL-MLLM/CoIN/checkpoints/LLaVA/CoIN/ind-8task/Task8_llava_lora_ours'

    # activation_8 = eval_model(args)
    # with open('activation_8.pkl', 'wb') as f:
    #     pickle.dump(activation_8, f)



    with open('activation_1.pkl', 'rb') as f:
        activation_1 = pickle.load(f)
    with open('activation_2.pkl', 'rb') as f:
        activation_2 = pickle.load(f)
    with open('activation_3.pkl', 'rb') as f:
        activation_3 = pickle.load(f)
    with open('activation_4.pkl', 'rb') as f:
        activation_4 = pickle.load(f)
    with open('activation_5.pkl', 'rb') as f:
        activation_5 = pickle.load(f)
    with open('activation_6.pkl', 'rb') as f:
        activation_6 = pickle.load(f)
    # with open('activation_7.pkl', 'rb') as f:
    #     activation_7 = pickle.load(f)
    # with open('activation_8.pkl', 'rb') as f:
    #     activation_8 = pickle.load(f)

    cka_sim_linear = []
    cka_sim_kernel = []

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_0_final_output'], activation_j['layer_0_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 0:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer0.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_1_final_output'], activation_j['layer_1_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 1:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer1.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_2_final_output'], activation_j['layer_2_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 2:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer2.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_3_final_output'], activation_j['layer_3_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 3:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer3.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_4_final_output'], activation_j['layer_4_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 4:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer4.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_5_final_output'], activation_j['layer_5_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 5:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer5.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_6_final_output'], activation_j['layer_6_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 6:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer6.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_7_final_output'], activation_j['layer_7_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 7:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer7.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_8_final_output'], activation_j['layer_8_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 8:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer8.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_9_final_output'], activation_j['layer_9_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 9:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer9.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_10_final_output'], activation_j['layer_10_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 10:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer10.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_11_final_output'], activation_j['layer_11_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 11:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer11.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_12_final_output'], activation_j['layer_12_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 12:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer12.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_13_final_output'], activation_j['layer_13_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 13:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer13.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_14_final_output'], activation_j['layer_14_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 14:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer14.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_15_final_output'], activation_j['layer_15_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  

    print('layer 15:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer15.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_16_final_output'], activation_j['layer_16_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 16:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer16.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_17_final_output'], activation_j['layer_17_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 17:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer17.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_18_final_output'], activation_j['layer_18_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 18:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer18.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_19_final_output'], activation_j['layer_19_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 19:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer19.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_20_final_output'], activation_j['layer_20_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 20:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer20.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_21_final_output'], activation_j['layer_21_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 21:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer21.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_22_final_output'], activation_j['layer_22_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 22:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer22.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_23_final_output'], activation_j['layer_23_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 23:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer23.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_24_final_output'], activation_j['layer_24_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 24:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer24.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_25_final_output'], activation_j['layer_25_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 25:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer25.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_26_final_output'], activation_j['layer_26_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 26:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer26.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_27_final_output'], activation_j['layer_27_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 27:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer27.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_28_final_output'], activation_j['layer_28_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 28:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer28.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_29_final_output'], activation_j['layer_29_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 29:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer29.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_30_final_output'], activation_j['layer_30_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  
    print('layer 30:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer30.png') 

    kernel_CKA_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i <= j:
                activation_i = locals()[f'activation_{i+1}']
                activation_j = locals()[f'activation_{j+1}']
                
                kernel_result = kernel_CKA(activation_i['layer_31_final_output'], activation_j['layer_31_final_output'])
                
                kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]  

    print('layer 31:')
    print(kernel_CKA_matrix)
    
    plt.figure(figsize=(10, 8), dpi=500)  
    sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)  
    plt.savefig('kernel-CKA-layer31.png') 
    #     activations_1_value = activation_1[layer_name] 
    #     activations_2_value = activation_2[layer_name]

    #     linear_result = linear_CKA(activations_1_value, activations_2_value)
    #     kernel_result = kernel_CKA(activations_1_value, activations_2_value)

    #     cka_sim_linear.append(np.round(linear_result, 4))
    #     cka_sim_kernel.append(np.round(kernel_result, 4))
    
    # print(cka_sim_linear)
    # print(cka_sim_kernel)
