import torch
import transformers
from transformers import AutoConfig
transformers.logging.set_verbosity_error()

from llavavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llavavid.conversation import conv_templates, SeparatorStyle
from llavavid.model.builder import load_pretrained_model
from llavavid.utils import disable_torch_init
from llavavid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import os
import json
import math
import time
import random
import argparse
import logging
import warnings
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu

# SET GPU 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# WARNINGS IGNORE
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# SEED
def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# PARSER
def parse_args():
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--model-path", type=str, default="lmms-lab/LLaVA-NeXT-Video-7B-DPO")
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=2)
    parser.add_argument("--for_get_frames_num", type=int, default=40)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--data_path", default="raw_data/sample_data", type=str) 
    parser.add_argument("--vqa_save_path", default="database/vqa", type=str) 
    return parser.parse_args()

def load_video(video_path, args):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, args.for_get_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

def run_inference(args, video_path):
    '''
    PROMPT EXPERIMENTS
    '''
    ## JY Vandalism
    # question = "Pay attention to any individuals who may be engaging in vandalism, such as damaging or defacing property. Identify and explain any significant, abnormal, or criminal situations observed in the video."
    
    ## base question
    # question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes"
    # question = "What does this video describe? A. Buiding B.Forest C.coutryside D.Moon \nAnswer with the option's letter from the given choices directly."
    
    ## TH question
    # action = "shoplifting" # Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Road Accident, Robbery, Shooting, Shoplifting, Stealing, Vandalism
    # question = f"The video includes scenes of {action}. Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes"
    
    # Warnings ignore
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)

    # Set model configuration parameters if they exist
    if args.overwrite == True:
        overwrite_config = {}
        overwrite_config["mm_resampler_type"] = args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
        overwrite_config["patchify_video_feature"] = False
        overwrite_config["mlp_bias"] = False ## 추가함.
        cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

        if "224" in cfg_pretrained.mm_vision_tower:
            # suppose the length of text tokens is around 1000, from bo's report
            least_token_number = args.for_get_frames_num*(16//args.mm_spatial_pool_stride)**2 + 1000
        else:
            least_token_number = args.for_get_frames_num*(24//args.mm_spatial_pool_stride)**2 + 1000

        scaling_factor = math.ceil(least_token_number/4096)

        if scaling_factor >= 2:
            if "mistral" not in cfg_pretrained._name_or_path.lower() and "7b" in cfg_pretrained._name_or_path.lower():
                overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
            overwrite_config["max_sequence_length"] = 4096 * scaling_factor
            overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config)
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    sample_set = {}
    question = "Describe the main subjects in the video, detailing their appearance and actions. Specifically, focus on any instances involving firearms, physical assault, or criminal activities. Additionally, identify and explain any significant, abnormal, or criminal situations observed in the video. Highlight any activities or behaviors that may be indicative of such situations, and provide detailed descriptions to clarify these observations."
    sample_set["Q"] = question
    sample_set["video_name"] = args.video_path

    # Check if the video exists
    if os.path.exists(video_path):
        video = load_video(video_path, args)
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
        video = [video]

    # Run inference on the video and add the output to the list
    qs = question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    cur_prompt = question
    with torch.inference_mode():
        model.update_prompt([[cur_prompt]])
        start_time = time.time()
        output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
        end_time = time.time()
        print(f"Time taken for inference: {end_time - start_time} seconds")

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(f"Question: {prompt}\n")
    print(f"Response: {outputs}\n")

    # Output save
    cwd = os.getcwd()
    save_path = os.path.join(cwd, args.vqa_save_path)
    start_index = args.video_path.rfind('/') + 1
    end_index = args.video_path.rfind('.')
    vid_name = args.video_path[start_index:end_index]

    file_path = f"{save_path}/{vid_name}.txt" 
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write("# Video MetaData\n")
        file.write(f"## Video Name: {vid_name}\n")
        file.write(f"## Video Path: {args.video_path}\n")
        file.write("\n")
        file.write("# LMM \n")
        file.write(f"## Model: {args.model_path} \n")
        file.write(f'## Query: {question}\n')
        file.write(f'## Tel-LLaVA Answer: {outputs}\n') 
        file.write(f'## Time taken for inference: {end_time - start_time} seconds\n') 

    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()

if __name__ == "__main__":
    set_seeds()
    args = parse_args()
    run_inference(args)