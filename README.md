# Multiplexer

## Installation

### Requirements:

```bash
conda create --name multiplexer
conda activate multiplexer
pip install yacs==0.1.8  # Note: conda only has yacs v0.1.6 now
pip install numpy
pip install black
pip install isort==5.9.3
pip install flake8==3.9.2
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge
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