## Get Started

``` bash
conda create -n ViDove python=3.10 -y
conda activate ViDove
pip install --upgrade pip
pip install -r requirements.txt
```


## Run

__batch processing recursive__
Evaluate all the videos in a folder(including subfolders)
```bash
python evaluation/recursive_parallel_translate.py --input_folder "path/to/your/videos/folder" --output_path "path/to/your/target/output/folder" --use_audio --parallel_workers 10 --shared_model
```

Evaluate a single video file
``` bash
python evaluation/single_eval.py --video_path "path/to/your/video" --output_path "path/to/your/output/folder"
```
