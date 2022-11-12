# Multiplexer

## Installation

### Requirements:

```bash
conda create --name multiplexer
conda activate multiplexer
# sudo apt install nvidia-cuda-toolkit  # install nvcc if it's not already there
pip install yacs==0.1.8  # Note: conda only has yacs v0.1.6 now
pip install numpy
pip install opencv-python
# run `nvcc --version` to decide the cudatoolkit version

conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install pyclipper
conda install shapely
conda install -c conda-forge pycocotools
conda install -c conda-forge ftfy
pip install tensorboard
pip install submitit
pip install tqdm
pip install editdistance
pip install scipy
pip install black
pip install isort==5.9.3
pip install flake8==3.9.2

python setup.py build_ext install
# Note: if the above command doesn't work,
# you can try the following depending on your 
# CUDA/nvcc/gcc compatibility versions and locations
# See https://stackoverflow.com/a/46380601
# For example, if `nvcc --version` says version 10.1, you can use/install g++-8 if it's not there
sudo apt install g++-8

python setup.py build develop

```

## Demo 
Activate the multiplexer environment if you haven't already done so:
```
conda init bash
source ~/.bashrc
conda activate multiplexer
```
Then, you can run the demo script for a single image inference by 
```
weight=YOUR_PATH_TO_WEIGHT_FILE
img=YOUR_PATH_TO_IMAGE_FILE
output=YOUR_OUTPUT_FILE_NAME

python -m demo.demo \
--config-file configs/demo.yaml \
--input $img \
--output $output \
MODEL.WEIGHT $weight
```

Alternatively, you can check out our [demo notebook](https://github.com/facebookresearch/MultiplexedOCR/blob/main/notebook/inference/demo.ipynb) for a quick demo!

## Training
Activate the multiplexer environment if you haven't already done so:
```
conda init bash
source ~/.bashrc
conda activate multiplexer
```
Then, modify the yaml file so that CHAR_MAP.DIR is pointing to the directory containing the character map jsons (examples could be found under [charmap/public/v3](https://github.com/facebookresearch/MultiplexedOCR/tree/main/charmap/public/v3)), and run the training/finetuning
```
yaml=PATH_TO_YAML_FILE
python3 tools/train_net.py --config-file $yaml
```

## Models

<table>
    <tr>
        <th>Model Name</th>
        <th>Download Link</th>
    </tr>
    <tr>
        <td>PMLX1G</td>
        <td><a href='https://dl.fbaipublicfiles.com/MultiplexedOCR/weights/PMLX1G/PMLX1G.pth'>weights</a>&nbsp;|&nbsp;<a href='https://dl.fbaipublicfiles.com/MultiplexedOCR/weights/PMLX1G/config.yaml'>config</a></td>
    </tr>
</table>

## Evaluation on MLT19

### Step 1: generate intermediate results
```
train_flow=PMLX1G_public
yaml_dir=aws/weights/PMLX1G
model_name=PMLX1G
min_size_test=2000
max_size_test=2560

python3 tools/launch_test.py \
--dataset mlt19 \
--name pmlx_test_$train_flow \
--yaml_dir $yaml_dir \
--yaml config.yaml \
INPUT.MIN_SIZE_TEST $min_size_test \
INPUT.MAX_SIZE_TEST $max_size_test \
MODEL.WEIGHT ${yaml_dir}/${model_name}.pth \
TEST.VIS False \
OUTPUT_DIR /checkpoint/$USER/flow/multiplexer/test/$train_flow \
OUTPUT.MLT19.TASK1 False \
OUTPUT.MLT19.TASK3 False \
OUTPUT.MLT19.TASK4 False \
OUTPUT.MLT19.VALIDATION_EVAL False \
OUTPUT.MLT19.INTERMEDIATE True \
OUTPUT.MLT19.INTERMEDIATE_WITH_PKL False \
OUTPUT.ZIP_PER_GPU True
```
### Step 2: generate files for submission

#### Task 4
```
train_flow=PMLX1G_public
model_name=PMLX1G
min_size_test=2000
max_size_test=2560
confidence_type=det
char_map_version=none
lexicon=none
score_det=0.2
score_rec_seq=0.8

python3 tools/launch_eval.py \
--name multiplexer_mlt19_test_${task}_${train_flow} \
--run_type local \
==cache_dir "/tmp/$USER/mlt19_test/${train_flow}/cache_files/" \
==char_map_version $char_map_version \
==confidence_type $confidence_type \
==intermediate_results /checkpoint/$USER/flow/multiplexer/test/${train_flow}/inference/mlt19_test/${model_name}_mlt19_intermediate.zip \
==protocol intermediate \
==lexicon $lexicon \
==score_det $score_det \
==score_rec_seq $score_rec_seq \
==seq on \
==split test \
==task task4 \
==zip_per_gpu
```

## Relationship to Mask TextSpotter v3

This project is under a lincense of Creative Commons Attribution-NonCommercial 4.0 International. Part of the code is inherited from [Mask TextSpotter v3](https://github.com/MhLiao/MaskTextSpotterV3), which is under the same license.

## Citing Multiplexer

If you use Multiplexer in your research or wish to refer to the baseline results, please use the following BibTeX entry.

```BibTeX
@inproceedings{huang2021multiplexed,
  title={A multiplexed network for end-to-end, multilingual ocr},
  author={Huang, Jing and Pang, Guan and Kovvuri, Rama and Toh, Mandy and Liang, Kevin J and Krishnan, Praveen and Yin, Xi and Hassner, Tal},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4547--4557},
  year={2021}
}
```