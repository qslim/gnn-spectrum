# A New Perspective on the Effects of Spectrum in Graph Neural Networks

[![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

This is the code of the paper "A New Perspective on the Effects of Spectrum in Graph Neural Networks".

## Requirements

The code is built upon [Deep Graph Library](https://www.dgl.ai/).

The following packages need to be installed:

- `pytorch==1.8.0`
- `dgl==0.7.0`
- `torch-geometric==1.7.1`
- `ogb==1.3.1`
- `numpy`
- `easydict`
- `tensorboard`
- `tqdm`
- `json5`

## Usage

#### ZINC
- Change directory to [zinc](zinc).
- Set hyper-parameters in [ZINC.json](zinc/ZINC.json).
- Run the script: `sh run_script.sh`

#### OGBG-MolPCBA
- Change directory to [ogbg/mol](ogbg/mol).
- Set hyper-parameters in [ogbg-molpcba.json](ogbg/mol/ogbg-molpcba.json).
- Run the script: `sh run_script.sh`

#### TUDataset
- Change directory to [tu](tu).
- Set dataset name in [run_script.sh](tu/run_script.sh) and set hyper-parameters in [configs/\<dataset\>.json](tu/configs).
- Run the script: `sh run_script.sh`

## Reference
```
@inproceedings{yang2022spectrum,
  title = {A New Perspective on the Effects of Spectrum in Graph Neural Networks},
  author = {Mingqi Yang and Yanming Shen and Rui Li and Heng Qi and Qiang Zhang and Baocai Yin},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  year = {2022},
}
```


## License

[MIT License](LICENSE)
