from vqa import *
from retrieval import *

def parse_args():
    parser = argparse.ArgumentParser()

    ## DATA & MODEL
    parser.add_argument("--data_path", default="raw_data/sample_data", type=str)
    parser.add_argument("--save_path", default="database/extracted_sample_data/", type=str)
    parser.add_argument("--model_name", default="microsoft/xclip-base-patch16-16-frames", type=str)
    
    ## SEARCH 
    parser.add_argument('--search_query', default="arson", type=str)
    
    ## MODEL PARAMETER
    parser.add_argument('--clip_len', default=16, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--top_k', default=3, type=int)

    # LLaVa
    parser.add_argument("--model-path", type=str, default="lmms-lab/LLaVA-NeXT-Video-7B-DPO")
    parser.add_argument("--video_path", type=str, default='/home/workspace/TelVid/raw_data/sample_data/03_road accident.mp4') # 삭제 
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
    parser.add_argument("--vqa_save_path", default="database/vqa", type=str) 
    return parser.parse_args()

if __name__ == "__main__":
    seed_everything() # SEED
    args = parse_args() # Args 
    video_extracter(args) # Video extracter 
    top_k_list, video_path = video_retrievel(args) # Video retrievel
    run_inference(args, video_path) # VQA