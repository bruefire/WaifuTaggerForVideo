# WDTaggerForVideo
This script performs batch tagging with [WD14Tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger) for multiple videos and outputs the results in either .txt files or a single json file.  
Output .txt files or single .json file can be used on [Text-To-Video Finetuning](https://github.com/ExponentialML/Text-To-Video-Finetuning) for ModelScope.  
The core part of this script is based on the one of scripts from [Kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts).  
I have just added the video generation part of the process.   

## Install
```
git clone https://github.com/bruefire/WaifuTaggerForVideo.git
cd WaifuTaggerForVideo
pip install -r requirements.txt
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

## Usage
```
# use .txt file format for getting result
python tagging.py --batch_size 20 /path/to/your/video/directory/ --clip_num 8

# use json format, and add manual tags
python tagging.py --batch_size 20 /path/to/your/video/directory/ --clip_num 8 --json output_name.json --tags "additional, manual, tag"
```
"clip_num" specifies the maximum number of frames for tagging per video.  
If you encounter VRAM shortage, please decrease the value of "batch_size".  
jsonとして出力する場合は--jsonを指定します  

## Output file(s) formats
### .txt
.txtファイルとして出力する場合は1つのファイルに対して1つのタグファイルが生成されます。  
また、一つのファイルの中にクリップされたフレームすべてに対するタグが含まれています。  
(タグの取捨選択方法は改善の余地があると思います)  
以下のような内容が書き込まれているはずです。
```
1girl, solo, looking_at_viewer, bangs, blue_eyes, closed_mouth, blue_hair, long_hair, shirt, gloves, holding, ...
```

### .json
jsonファイルとして出力する場合は全体の結果をまとめた単一の.jsonが出力されます。  
以下のような内容となっているはずです。  
```
{
    "name": "My Videos",
    "data": [
        {
            "video_path": "E:\\userdata\\videos\\ai_trainning\\t2v-v2\\video1.mp4",
            "num_frames": 73,
            "data": [
                {
                    "frame_index": 0,
                    "prompt": "1girl, solo, looking_at_viewer, bangs, blue_eyes, closed_mouth, blue_hair, ..."
                },
                {
                    "frame_index": 18,
                    "prompt": "1girl, solo, long_hair, looking_at_viewer, bangs, blue_eyes, shirt, closed_mouth, ..."
                },
                {
                    "frame_index": 27,
                    "prompt": "1girl, solo, long_hair, bangs, shirt, gloves, holding, closed_mouth, blue_hair, ..."
                }
            ]
        },
        {
            "video_path": "E:\\userdata\\videos\\ai_trainning\\t2v-v2\\video2.mp4",
            "num_frames": 12,
            "data": [
				...
            ]
        },
		...
	]
}
```
