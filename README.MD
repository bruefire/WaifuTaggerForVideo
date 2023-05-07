# WaifuTaggerForVideo
This script performs batch tagging with [WD14Tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger) for multiple videos, and outputs the results in either CSV or JSON.   
The output result file(s) can be used with [Text-To-Video Finetuning](https://github.com/ExponentialML/Text-To-Video-Finetuning) for ModelScope.  
  
This script has been tested with Python 3.9.13 on Windows.  
The core part of this script is based on the one of the scripts from [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts).  
I have just added the video generation part of the process.   

## Output formats
### Single line CSV (.txt)
When getting results with CSV files, one tag file is generated for each video file.  
Also each file contains tags for all frames that have been clipped.  
(There is room for improvement in the selection of which tags to keep)  
The contents of the .txt file with the same name as the video file should be written as follows:
```
1girl, solo, looking_at_viewer, bangs, blue_eyes, closed_mouth, blue_hair, long_hair, shirt, gloves, holding, ...
```

### JSON (.json)
When outputting the results with JSON format, a single file containing the aggregated results will be generated.  
The content should look like the following (on Windows):  
```
{
    "name": "My Videos",
    "data": [
        {
            "video_path": "C:\\userdata\\videos\\ai_trainning\\t2v-v2\\video1.mp4",
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

## Install
```
git clone https://github.com/bruefire/WaifuTaggerForVideo.git
cd WaifuTaggerForVideo
pip install -r requirements.txt
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

## Usage
```
# use single line CSV format for getting results
python tagging.py /path/to/video/directory/ --batch_size 20 --clip_num 8

# use JSON format
python tagging.py /path/to/video/directory/ --batch_size 20 --clip_num 8 --json result_file.json
```
* "clip_num" specifies the maximum number of frames for tagging per video.  
* If you encounter VRAM shortage, please decrease the value of "batch_size".   
* If specific tags are required, please add the option as follows: "--tags 'manual-tag1, another-one'".  
