
# ChangeBind: A Hybrid Change Encoder for Remote Sensing Change Detection

This repo contains the official **PyTorch** code for ChangeBind.

Introduction
-----------------
ChangeBind utilizes a change encoder that leverages local and global feature representations to capture both subtle and large change feature information to precisely estimate the change regions.

### :point_right: Data structure

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
@inproceedings{changebind2024,
  title={ChangeBind: A Hybrid Change Encoder for Remote Sensing Change Detection},
  author={Noman, Mubashir and Fiaz, Mustansar and Cholakkal, Hisham},
  booktitle={IGARSS},
  year={2024}
}
```

## Acknowledgements
Thanks to the codebases [[ScratchFormer]](https://github.com/mustansarfiaz/ScratchFormer) [[BIT]](https://github.com/justchenhao/BIT_CD) [[ChangeFormer]](https://github.com/wgcban/ChangeFormer). 

## See Also
[ScratchFormer](https://github.com/mustansarfiaz/ScratchFormer): Remote Sensing Change Detection With Transformers Trained from Scratch

[ELGCNet](https://github.com/techmn/elgcnet): Efficient Local-Global Context Aggregation for Remote Sensing Change Detection
