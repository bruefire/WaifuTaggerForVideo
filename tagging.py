import argparse
import csv
import glob
import os
import shutil
import json

from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
import torch
import ffmpeg
import datetime


# from wd14 tagger
IMAGE_SIZE = 448

# wd-v1-4-swinv2-tagger-v2 / wd-v1-4-vit-tagger / wd-v1-4-vit-tagger-v2/ wd-v1-4-convnext-tagger / wd-v1-4-convnext-tagger-v2
DEFAULT_WD14_TAGGER_REPO = 'SmilingWolf/wd-v1-4-convnext-tagger-v2'
FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]
CSV_FILE = FILES[-1]
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
VIDEO_EXTENSIONS = [".mp4"]


class VideoInfoJson():

  def __init__(self, out_path, name="My Videos"):
    self.out_path = out_path
    self.name = name
    self.data = {}
    
  def __len__(self):
    return len(self.data)
    
  def __getitem__(self, key):
    img_path = self.data[key]
    
  def add_video(self, video_path, num_frames):
    self.data[video_path] = {}
    self.data[video_path]["num_frames"] = str(num_frames)
    self.data[video_path]["data"] = []
    
  def add_video_content(self, video_path, frame_index, taginfo):
    new_content = {}
    new_content["frame_index"] = str(frame_index)
    new_content["prompt"] = taginfo.strip()
    if video_path in self.data:
        self.data[video_path]["data"].append(new_content)
        
  def make(self):
    with open(self.out_path, 'w', encoding='utf-8') as f:
      result = '''{
    "name": "''' + self.name + '''",
    "data": ['''
      for vpath, info in self.data.items():
        result += '''
        {
            "video_path": ''' + json.dumps(vpath) + ''',
            "num_frames": ''' + info["num_frames"] + ''',
            "data": ['''
        for content in info["data"]:
          result += '''
                {
                    "frame_index": ''' + content["frame_index"] + ''',
                    "prompt": ''' + json.dumps(content["prompt"]) + '''
                },'''
        result = result[:-1]
        result += '''
            ]
        },'''
      result = result[:-1]
      result += '''
    ]
}
'''
      f.write(result)
    

def glob_images(directory, base="*"):
  img_paths = []
  for ext in IMAGE_EXTENSIONS:
    if base == '*':
      img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
    else:
      img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
  # img_paths = list(set(img_paths))                    # 重複を排除
  # img_paths.sort()
  return img_paths

def glob_videos(directory, base="*"):
  vid_paths = []
  for ext in VIDEO_EXTENSIONS:
    if base == '*':
      vid_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
    else:
      vid_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
      
  return vid_paths
  
def clip_videos(vid_paths, clip_num, json_obj):
  print("making video clips..")
  clip_paths = []
  
  # クリップを生成
  for vid_path in tqdm(vid_paths, smoothing=0.0):
    probe = ffmpeg.probe(vid_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    num_frames = int(video_info['nb_frames'])

    # クリップ用フォルダ作成
    clip_dir_name = os.path.splitext(os.path.basename(vid_path))[0]
    input_clip_dir = os.path.join(os.path.dirname(vid_path), clip_dir_name)
    if os.path.exists(input_clip_dir):
      shutil.rmtree(input_clip_dir)
    os.makedirs(input_clip_dir)

    # 指定枚数だけ抜き出して一時保存
    video = ffmpeg.input(vid_path)
    for frame in range(clip_num):
      frame_num = min(int((num_frames / clip_num) * frame), num_frames - 1)
      output_path = os.path.join(input_clip_dir, str(frame_num) + '.png')
      clip_image = ffmpeg.output(video, output_path, vf='select=eq(n\,'+ str(frame_num) +')', vframes=1)
      ffmpeg.run(clip_image, quiet=True, overwrite_output=True)

      clip_paths.append(output_path)
      
    #json作成の場合 対象ビデオ要素作成
    if json_obj is not None:
      json_obj.add_video(vid_path, num_frames)
      
  return clip_paths

def integrate_video_tags(video_paths, json_obj):
  for video_path in video_paths:
    target_dir = os.path.splitext(video_path)[0]
    # フォルダ内のテキストファイルをリストで取得
    txt_files = [f for f in os.listdir(target_dir) if f.endswith('.txt')]
    
    if json_obj is None:
      # ファイル内容を抽出し、1つのリストにまとめる
      lines = []
      for file in txt_files:
          with open(os.path.join(target_dir, file), 'r') as f:
              lines += [line.strip() for line in f.readlines()]
      # 改行を","に変換
      lines_t = []
      for line in lines:
        lines_t.extend(line.split(','))
      lines = lines_t
      # 行頭行末の空白をトリム
      lines = [line.strip() for line in lines]
      # 行をソートし、重複行を削除
      lines = set(lines)
      # 各行を","で連結して1つの行にする
      result = ', '.join(lines)
      # ファイル出力
      print(target_dir)
      with open(target_dir + '.txt', 'w') as f:
          f.write(result)
    else:
      # 各フレーム内容を追加
      for file in txt_files:
          with open(os.path.join(target_dir, file), 'r') as f:
            frame_index = os.path.splitext(os.path.basename(file))[0]
            taginfo = f.readlines()[0]
            json_obj.add_video_content(video_path, frame_index, taginfo)
      
    # クリップフォルダ削除
    if os.path.exists(target_dir):
      shutil.rmtree(target_dir)
      
  if json_obj is not None:
    json_obj.make()
    

def preprocess_image(image):
  image = np.array(image)
  image = image[:, :, ::-1]                         # RGB->BGR

  # pad to square
  size = max(image.shape[0:2])
  pad_x = size - image.shape[1]
  pad_y = size - image.shape[0]
  pad_l = pad_x // 2
  pad_t = pad_y // 2
  image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode='constant', constant_values=255)

  interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
  image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)

  image = image.astype(np.float32)
  return image


class ImageLoadingPrepDataset(torch.utils.data.Dataset):
  def __init__(self, image_paths):
    self.images = image_paths

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    img_path = self.images[idx]

    try:
      image = Image.open(img_path).convert("RGB")
      image = preprocess_image(image)
      tensor = torch.tensor(image)
    except Exception as e:
      print(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
      return None

    return (tensor, img_path)


def collate_fn_remove_corrupted(batch):
  """Collate function that allows to remove corrupted examples in the
  dataloader. It expects that the dataloader returns 'None' when that occurs.
  The 'None's in the batch are removed.
  """
  # Filter out all the Nones (corrupted examples)
  batch = list(filter(lambda x: x is not None, batch))
  return batch


def main(args):
  # hf_hub_downloadをそのまま使うとsymlink関係で問題があるらしいので、キャッシュディレクトリとforce_filenameを指定してなんとかする
  # depreacatedの警告が出るけどなくなったらその時
  # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/issues/22
  if not os.path.exists(args.model_dir) or args.force_download:
    print(f"downloading wd14 tagger model from hf_hub. id: {args.repo_id}")
    for file in FILES:
      hf_hub_download(args.repo_id, file, cache_dir=args.model_dir, force_download=True, force_filename=file)
    for file in SUB_DIR_FILES:
      hf_hub_download(args.repo_id, file, subfolder=SUB_DIR, cache_dir=os.path.join(
          args.model_dir, SUB_DIR), force_download=True, force_filename=file)
  else:
    print("using existing wd14 tagger model")

  # 画像を読み込む
  image_paths = glob_images(args.train_data_dir)
  print(f"found {len(image_paths)} images.")

  # 動画を読み込む
  json_obj = None
  if args.json:
    json_obj = VideoInfoJson(args.json)
    print("make a video json file.")
  
  video_paths = glob_videos(args.train_data_dir)
  print(f"found {len(video_paths)} videos.")
  
  clip_paths = clip_videos(video_paths, args.clip_num, json_obj)
  image_paths.extend(clip_paths)
  print(f"{len(clip_paths)} frames clipped from videos.")

  print("loading model and labels")
  model = load_model(args.model_dir)
  print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

  # label_names = pd.read_csv("2022_0000_0899_6549/selected_tags.csv")
  # 依存ライブラリを増やしたくないので自力で読むよ
  with open(os.path.join(args.model_dir, CSV_FILE), "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    l = [row for row in reader]
    header = l[0]             # tag_id,name,category,count
    rows = l[1:]
  assert header[0] == 'tag_id' and header[1] == 'name' and header[2] == 'category', f"unexpected csv format: {header}"

  tags = [row[1] for row in rows[1:] if row[2] == '0']      # categoryが0、つまり通常のタグのみ

  # 推論する
  def run_batch(path_imgs):
    imgs = np.array([im for _, im in path_imgs])

    probs = model(imgs, training=False)
    probs = probs.numpy()

    for (image_path, _), prob in zip(path_imgs, probs):
      # 最初の4つはratingなので無視する
      # # First 4 labels are actually ratings: pick one with argmax
      # ratings_names = label_names[:4]
      # rating_index = ratings_names["probs"].argmax()
      # found_rating = ratings_names[rating_index: rating_index + 1][["name", "probs"]]

      # それ以降はタグなのでconfidenceがthresholdより高いものを追加する
      # Everything else is tags: pick any where prediction confidence > threshold
      tag_text = ""
      for i, p in enumerate(prob[4:]):                # numpyとか使うのが良いけど、まあそれほど数も多くないのでループで
        if p >= args.thresh and i < len(tags):
          tag_text += ", " + tags[i]

      if len(tag_text) > 0:
        tag_text = tag_text[2:]                   # 最初の ", " を消す

      with open(os.path.splitext(image_path)[0] + args.caption_extension, "wt", encoding='utf-8') as f:
        tag_text = args.tags + ", " + tag_text if args.tags else tag_text   # マニュアルタグ
        f.write(tag_text + '\n')
        if args.debug:
          print(image_path, tag_text)

  # 読み込みの高速化のためにDataLoaderを使うオプション
  if args.max_data_loader_n_workers is not None:
    dataset = ImageLoadingPrepDataset(image_paths)
    data = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.max_data_loader_n_workers, collate_fn=collate_fn_remove_corrupted, drop_last=False)
  else:
    data = [[(None, ip)] for ip in image_paths]

  b_imgs = []
  for data_entry in tqdm(data, smoothing=0.0):
    for data in data_entry:
      if data is None:
        continue

      image, image_path = data
      if image is not None:
        image = image.detach().numpy()
      else:
        try:
          image = Image.open(image_path)
          if image.mode != 'RGB':
            image = image.convert("RGB")
          image = preprocess_image(image)
        except Exception as e:
          print(f"Could not load image path / 画像を読み込めません: {image_path}, error: {e}")
          continue
      b_imgs.append((image_path, image))

      if len(b_imgs) >= args.batch_size:
        run_batch(b_imgs)
        b_imgs.clear()

  if len(b_imgs) > 0:
    run_batch(b_imgs)

  # 各動画クリップの推論結果をまとめる
  integrate_video_tags(video_paths, json_obj)

  print("done!")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
  parser.add_argument("--repo_id", type=str, default=DEFAULT_WD14_TAGGER_REPO,
                      help="repo id for wd14 tagger on Hugging Face / Hugging Faceのwd14 taggerのリポジトリID")
  parser.add_argument("--model_dir", type=str, default="wd14_tagger_model",
                      help="directory to store wd14 tagger model / wd14 taggerのモデルを格納するディレクトリ")
  parser.add_argument("--force_download", action='store_true',
                      help="force downloading wd14 tagger models / wd14 taggerのモデルを再ダウンロードします")
  parser.add_argument("--thresh", type=float, default=0.35, help="threshold of confidence to add a tag / タグを追加するか判定する閾値")
  parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
  parser.add_argument("--max_data_loader_n_workers", type=int, default=None,
                      help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）")
  parser.add_argument("--caption_extention", type=str, default=None,
                      help="extension of caption file (for backward compatibility) / 出力されるキャプションファイルの拡張子（スペルミスしていたのを残してあります）")
  parser.add_argument("--caption_extension", type=str, default=".txt", help="extension of caption file / 出力されるキャプションファイルの拡張子")
  parser.add_argument("--debug", action="store_true", help="debug mode")
  parser.add_argument("--clip_num", type=int, default=3, help="number of clipping video")
  parser.add_argument("--json", type=str, default="", help="make json file for t2v fine-tuning")
  parser.add_argument("--tags", type=str, default="", help="add tags manually")

  args = parser.parse_args()

  # スペルミスしていたオプションを復元する
  if args.caption_extention is not None:
    args.caption_extension = args.caption_extention

  main(args)
