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
You can run a demo script for a single image inference by 
```
python -m demo.demo --config-file configs/seg_rec_poly_fuse_feature_once.yaml
```
The following is to be supported (fixing the \_BASE\_ field in config)
```
python -m demo.demo --config-file configs/multi_seq_lang_v2.yaml
```

## Training

```
conda init bash
source ~/.bashrc
conda activate multiplexer
python3 tools/train_net.py --config-file /checkpoint/jinghuang/multiplexer/configs/multiplexer_v1.yaml
```

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