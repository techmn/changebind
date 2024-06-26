
# ChangeBind: A Hybrid Change Encoder for Remote Sensing Change Detection

This repo contains the official **PyTorch** code for ChangeBind [[Arxiv]](https://arxiv.org/abs/2404.17565).

## Introduction
ChangeBind utilizes a change encoder that leverages local and global feature representations to capture both subtle and large change feature information to precisely estimate the change regions.

### :arrow_right: Requirements
```
pytorch 1.10.0
timm 0.4.12
opencv-python
tqdm
pillow
```

### :arrow_right: Data structure

```
"""
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
"""
```

`A`: images at time frame t1;

`B`:images at time frame t2;

`label`: label masks;

`list`: contains `train.txt, val.txt and test.txt`, each file records the image names (____.png) in the change detection dataset.


## Citation
```
@misc{changebind2024,
  title={ChangeBind: A Hybrid Change Encoder for Remote Sensing Change Detection}, 
  author={Mubashir Noman and Mustansar Fiaz and Hisham Cholakkal},
  year={2024},
  eprint={2404.17565},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
  url={https://arxiv.org/abs/2404.17565}, 
}
```

## Acknowledgements
Thanks to the codebases [[ScratchFormer]](https://github.com/mustansarfiaz/ScratchFormer) [[BIT]](https://github.com/justchenhao/BIT_CD) [[ChangeFormer]](https://github.com/wgcban/ChangeFormer). 

## See Also
[ScratchFormer](https://github.com/mustansarfiaz/ScratchFormer): Remote Sensing Change Detection With Transformers Trained from Scratch

[ELGCNet](https://github.com/techmn/elgcnet): Efficient Local-Global Context Aggregation for Remote Sensing Change Detection
