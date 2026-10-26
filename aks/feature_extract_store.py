import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from transformers import CLIPProcessor, CLIPModel
import json
from decord import VideoReader
from decord import cpu
import numpy as np
import os
import logging
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract Video Feature')

    parser.add_argument(
        '--dataset_name',
        type=str,
        default='Charades-CD',
        help='support Charades-CD, ActivityNet-CD, Charades_STA and ActivityNet'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='./datasets/Charades-CD',
        help='path of the annotation dataset folder'
    )
    parser.add_argument(
        '--extract_feature_model',
        type=str,
        default='blip',
        help='blip/clip'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='./outscores',
        help='path of output scores and frames'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda'
    )

    return parser.parse_args()


def get_video_file(video_dir, video_key):
    for suffix in ['.mp4', '.mkv', '.avi']:
        video_path = os.path.join(video_dir, f"{video_key}{suffix}")
        if os.path.exists(video_path):
            return video_path


def atomic_json_dump(obj, path):

    tmp_path = path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_json_if_exists(path, default):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return default


def export_current_results(
    processed_keys,
    scores_by_video,
    frames_by_video,
    fps_all,
    score_path,
    frame_path,
    fps_path,
    score_by_video_path,
    frame_by_video_path
):


    flat_scores = []
    flat_frames = []

    for key in processed_keys:
        if key in scores_by_video:
            flat_scores.extend(scores_by_video[key])
        if key in frames_by_video:
            flat_frames.extend(frames_by_video[key])

    atomic_json_dump(flat_scores, score_path)
    atomic_json_dump(flat_frames, frame_path)
    atomic_json_dump(fps_all, fps_path)

    atomic_json_dump(scores_by_video, score_by_video_path)
    atomic_json_dump(frames_by_video, frame_by_video_path)


def main(args):
    log_dir = os.path.join(args.output_file, args.dataset_name)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'extraction.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    if args.dataset_name == "Charades-CD":
        label_path = os.path.join(args.dataset_path, 'charades_train.json')
        video_path = os.path.join(args.dataset_path, 'videos')

    elif args.dataset_name == "ActivityNet-CD":
        label_path = os.path.join(args.dataset_path, 'anet_train.json')
        video_path = os.path.join(args.dataset_path, 'videos2')

    elif args.dataset_name == "Charades_STA":
        label_path = os.path.join(
            args.dataset_path,
            'glance_charades_train.json'
        )
        dataset_parent = os.path.dirname(os.path.normpath(args.dataset_path))
        video_path = os.path.join(
            dataset_parent,
            'Charades-CD',
            'videos'
        )

    elif args.dataset_name == "ActivityNet":
        label_path = os.path.join(
            args.dataset_path,
            'glance_train.json'
        )
        dataset_parent = os.path.dirname(os.path.normpath(args.dataset_path))
        video_path = os.path.join(
            dataset_parent,
            'ActivityNet-CD',
            'videos2'
        )

    else:
        raise ValueError(
            "dataset_name must be Charades-CD, ActivityNet-CD, "
            "Charades_STA or ActivityNet"
        )

    if not os.path.exists(label_path):
        raise OSError(f"the label file does not exist: {label_path}")

    with open(label_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    device = args.device

    if args.extract_feature_model == 'blip':
        model, vis_processors, text_processors = load_model_and_preprocess(
            "blip_image_text_matching",
            "large",
            device=device,
            is_eval=True
        )
    elif args.extract_feature_model == 'clip':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.to(device)
        model.eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    elif args.extract_feature_model == 'sevila':
        raise ValueError('Not support sevila')
    else:
        raise ValueError("model not support")

    out_score_path = os.path.join(
        args.output_file,
        args.dataset_name,
        args.extract_feature_model
    )
    os.makedirs(out_score_path, exist_ok=True)

    score_path = os.path.join(out_score_path, 'scores.json')
    frame_path = os.path.join(out_score_path, 'frames.json')
    fps_path = os.path.join(out_score_path, 'selected_fps.json')

    score_by_video_path = os.path.join(out_score_path, 'scores_by_video.json')
    frame_by_video_path = os.path.join(out_score_path, 'frames_by_video.json')

    checkpoint_path = os.path.join(out_score_path, 'checkpoint.json')
    failed_path = os.path.join(out_score_path, 'failed_videos.json')

    checkpoint = load_json_if_exists(
        checkpoint_path,
        {
            "processed_keys": [],
            "scores_by_video": {},
            "frames_by_video": {},
            "fps_all": {}
        }
    )

    processed_keys = checkpoint.get("processed_keys", [])
    scores_by_video = checkpoint.get("scores_by_video", {})
    frames_by_video = checkpoint.get("frames_by_video", {})
    fps_all = checkpoint.get("fps_all", {})

    failed_videos = load_json_if_exists(failed_path, {})

    processed_set = set(processed_keys)

    logging.info(f"总视频数: {len(datas)}")
    logging.info(f"已完成视频数: {len(processed_set)}")
    logging.info(f"结果输出目录: {out_score_path}")
    logging.info(f"标注文件: {label_path}")
    logging.info(f"视频目录: {video_path}")

    for idx, (key, value) in enumerate(datas.items()):
        if key in processed_set:
            logging.info(f"[跳过] 视频已处理: {key}")
            continue

        logging.info(f"[开始] 处理视频 {idx + 1}/{len(datas)}: {key}")

        try:
            text = value['sentences']

            video = get_video_file(video_path, key)

            vr = VideoReader(video, ctx=cpu(0), num_threads=1)

            fps = float(vr.get_avg_fps())
            fps_int = max(int(fps), 1)

            frame_nums = int(len(vr) / fps_int * 2)

            video_scores = []
            video_frames = []

            if args.extract_feature_model == 'blip':
                for item in text:
                    score = []
                    frame_num = []

                    txt = text_processors["eval"](item)

                    for j in range(frame_nums):
                        frame_idx = int(j * fps_int / 2)
                        frame_idx = min(frame_idx, len(vr) - 1)

                        raw_image = np.array(vr[frame_idx])
                        raw_image = Image.fromarray(raw_image)

                        img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

                        with torch.no_grad():
                            blip_output, blip_embedding = model(
                                {
                                    "image": img,
                                    "text_input": txt
                                },
                                match_head="itm-e"
                            )

                        blip_scores = torch.nn.functional.softmax(
                            blip_output,
                            dim=1
                        )

                        score.append(blip_scores[:, 1].item())
                        frame_num.append(frame_idx)

                    video_scores.append(score)
                    video_frames.append(frame_num)

            elif args.extract_feature_model == 'clip':
                for item in text:
                    score = []
                    frame_num = []

                    inputs_text = processor(
                        text=item,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(device)

                    with torch.no_grad():
                        text_features = model.get_text_features(**inputs_text)

                    for j in range(frame_nums):
                        frame_idx = int(j * fps_int / 2)
                        frame_idx = min(frame_idx, len(vr) - 1)

                        raw_image = np.array(vr[frame_idx])
                        raw_image = Image.fromarray(raw_image)

                        inputs_image = processor(
                            images=raw_image,
                            return_tensors="pt",
                            padding=True
                        ).to(device)

                        with torch.no_grad():
                            image_features = model.get_image_features(
                                **inputs_image
                            )

                        clip_score = torch.nn.CosineSimilarity(dim=-1)(
                            text_features,
                            image_features
                        )

                        score.append(clip_score.item())
                        frame_num.append(frame_idx)

                    video_scores.append(score)
                    video_frames.append(frame_num)

            else:
                raise ValueError('Not support sevila')


            scores_by_video[key] = video_scores
            frames_by_video[key] = video_frames
            fps_all[key] = fps

            processed_keys.append(key)
            processed_set.add(key)

            checkpoint = {
                "processed_keys": processed_keys,
                "scores_by_video": scores_by_video,
                "frames_by_video": frames_by_video,
                "fps_all": fps_all
            }


            atomic_json_dump(checkpoint, checkpoint_path)


            export_current_results(
                processed_keys,
                scores_by_video,
                frames_by_video,
                fps_all,
                score_path,
                frame_path,
                fps_path,
                score_by_video_path,
                frame_by_video_path
            )

            logging.info(
                f"[完成] 视频: {key}，句子数: {len(text)}，"
                f"采样帧数: {frame_nums}，已保存断点"
            )

        except Exception as e:
            logging.exception(
                f"[失败] 视频 {key} 处理失败，已记录，继续处理下一个视频"
            )

            failed_videos[key] = str(e)
            atomic_json_dump(failed_videos, failed_path)

            continue

    logging.info("全部可处理视频已完成")
    logging.info(f"最终 scores.json: {score_path}")
    logging.info(f"最终 frames.json: {frame_path}")
    logging.info(f"断点文件 checkpoint.json: {checkpoint_path}")
    logging.info(f"失败视频记录 failed_videos.json: {failed_path}")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)


