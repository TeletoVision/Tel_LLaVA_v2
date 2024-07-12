import os
import av
import json
import warnings

import torch
import numpy as np
from transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download

from argparse import ArgumentParser

'''
::: MODEL LIST :::

## Fully-supervised on Kinetics-400:
# model_name = "microsoft/xclip-base-patch32" # clip_len = 8
# model_name = "microsoft/xclip-base-patch32-16-frames" # clip_len = 16
# model_name = "microsoft/xclip-base-patch16" # clip_len = 8
# model_name = "microsoft/xclip-base-patch16-16-frames" # clip_len = 16
# model_name = "microsoft/xclip-large-patch14" # clip_len = 8
# model_name = "microsoft/xclip-large-patch14-16-frames" #  clip_len = 16

## Fully-supervised on Kinetics-600:
# model_name = "microsoft/xclip-base-patch16-kinetics-600" # clip_len = 8
# model_name = "microsoft/xclip-base-patch16-kinetics-600-16-frames" # clip_len = 16
# model_name = "microsoft/xclip-large-patch14-kinetics-600" # clip_len = 8

## HMDB-51
# model_name = "microsoft/xclip-base-patch16-hmdb-2-shot" # clip_len = 32
# model_name = "microsoft/xclip-base-patch16-hmdb-4-shot" # clip_len = 32
# model_name = "microsoft/xclip-base-patch16-hmdb-8-shot" # clip_len = 32
# model_name = "microsoft/xclip-base-patch16-hmdb-16-shot" # clip_len = 32

## UCF-101
# model_name = "microsoft/xclip-base-patch16-ucf-2-shot" # clip_len = 32
# model_name = "microsoft/xclip-base-patch16-ucf-4-shot" # clip_len = 32
# model_name = "microsoft/xclip-base-patch16-ucf-8-shot" # clip_len = 32
# model_name = "microsoft/xclip-base-patch16-ucf-16-shot" # clip_len = 32

## Kinetics-400
# model_name = "microsoft/xclip-base-patch16-zero-shot" # clip_len = 8
'''


def seed_everything(SEED=42):

    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def video_extracter(args):

    warnings.filterwarnings("ignore")
    seed_everything(args.seed)
    
    cwd = os.getcwd()
    data_path = os.path.join(cwd, args.data_path)
    save_path = os.path.join(cwd, args.save_path)
    video_path_list = os.listdir(data_path)
    # print(f"Video Path List: {video_path_list}")
    
    info = []
    video_vector_list = []

    for idx, file_path in enumerate(video_path_list):
        video_path = os.path.join(data_path, file_path)
        container = av.open(video_path)
        
        frame_sample_rate = int((container.streams.video[0].frames-1) / args.clip_len)
        
        # sample clip_len frames
        indices   = sample_frame_indices(
            clip_len=args.clip_len, frame_sample_rate=frame_sample_rate,
            seg_len=container.streams.video[0].frames
            )
        
        video     = read_video_pyav(container, indices)
        
        processor = AutoProcessor.from_pretrained(args.model_name).image_processor
        inputs    = processor(images=list(video), return_tensors="pt", padding=True)

        video_vector = inputs['pixel_values'].tolist()
        video_info = {
            "video_index":  idx,
            "video_name":   file_path[:-4],
            "video_path":   video_path,
        }

        info.append(video_info)
        video_vector_list.append(np.array(video_vector))
    
    with open(save_path + 'extracted_video.json', 'w') as f:
        json.dump(info, f, indent=4)
    
    np.save(save_path + 'extracted_video', video_vector_list)
    print(f"Processed {len(info)} videos.")


def video_retrievel(args):
    
    cwd = os.getcwd()
    save_path = os.path.join(cwd, args.save_path)
    
    video_vector_list = np.load(save_path + 'extracted_video.npy', allow_pickle=True)
    video_meta_data   = json.load(open(save_path + 'extracted_video.json', 'r'))

    search_query = args.search_query
    # search_query = map(lambda x : f"A video of, {x}", search_query)  ## TEXT TEMPLATE
    print(f'Search Query: {search_query}.')

    processor = AutoProcessor.from_pretrained(args.model_name).tokenizer
    model     = AutoModel.from_pretrained(args.model_name)
    inputs    = processor(text=search_query, return_tensors="pt", padding=True)
    
    similarity_list = []
    text_query = inputs.input_ids
    
    for i, video in enumerate(video_vector_list):
        model_inputs = {
            "input_ids": text_query,
            "pixel_values": torch.tensor(video)
        }
        
        with torch.no_grad():
            outputs = model(**model_inputs)
    
        logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
        similarity_list.append(float(logits_per_video[0][0]))

        print(f'{i+1}. similarity score : {float(logits_per_video[0][0])}')
        
    similarity_list = np.array(similarity_list)
    # similarity_list = np.sort(similarity_list)[::-1]
    top_k = np.argsort(similarity_list)[::-1][:args.top_k]  # top 3

    top_k_list = {
        'top_k_index': [],
        'similarity_score': [],
        'video_path': []
        }  ## 부가적인 메타 정보도 포함될 수 있기에 dict로 구성
    
    for video in video_meta_data:
        if video['video_index'] in top_k:
            top_k_index = np.where(top_k == video['video_index'])[0][0]
            top_k_list['top_k_index'].append(top_k_index)
            top_k_list['similarity_score'].append(similarity_list[video['video_index']])
            top_k_list['video_path'].append(video['video_path'])

    index = top_k_list['top_k_index'].index(1)
    video_path = top_k_list['video_path'][index]
    print(f'\nRetrieval Video is {video_path}\n')
    return top_k_list, video_path

if __name__ == "__main__":

    parser = ArgumentParser(description="Retrieval")

    ## DATA & MODEL
    parser.add_argument("--data_path", default="raw_data/sample_data", type=str)
    parser.add_argument("--save_path", default="database/extracted_sample_data/", type=str)
    parser.add_argument("--model_name", default="microsoft/xclip-large-patch14-kinetics-600", type=str)
    
    ## SEARCH 
    parser.add_argument('--search_query', default="arson", type=str)
    
    ## MODEL PARAMETER
    parser.add_argument('--clip_len', default=16, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--top_k', default=3, type=int)

    args = parser.parse_args()

    video_extracter(args)
    video_retrievel(args)
