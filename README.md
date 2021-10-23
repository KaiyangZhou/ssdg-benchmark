# Semi-Supervised Domain Generalization

This code is the official implementation of the following paper: [Semi-Supervised Domain Generalization with Stochastic StyleMatch](https://arxiv.org/abs/2106.00592). The paper addresses a practical and yet under-studied setting for domain generalization: one needs to use limited labeled data along with abundant unlabeled data gathered from multiple distinct domains to learn a generalizable model. This setting greatly challenges existing domain generalization methods, which are not designed to deal with unlabeled data and are thus less scalable in practice. Our approach, StyleMatch, extends the pseudo-labeling-based [FixMatch](https://arxiv.org/abs/2001.07685)—a state-of-the-art semi-supervised learning framework—in two crucial ways: 1) a stochastic classifier is designed to reduce overfitting and 2) the two-view consistency learning paradigm in FixMatch is upgraded to a multi-view version with style augmentation as the third complementary view. Two benchmarks are constructed for evaluation. Please see the paper at https://arxiv.org/abs/2106.00592 for more details.

# Updates

**23-10-2021**: StyleMatch has been accepted to [NeurIPS 2021 Workshop on Distribution Shifts: Connecting Methods and Applications](https://sites.google.com/view/distshift2021).

## How to setup the environment

This code is built on top of [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). Please follow the instructions provided in https://github.com/KaiyangZhou/Dassl.pytorch to install the `dassl` environment, as well as to prepare the datasets, PACS and OfficeHome. The five random labeled-unlabeled splits can be downloaded at the following links: [pacs](https://drive.google.com/file/d/1PSliZDI9D-_Wrr3tfRzGVtN2cpM1892p/view?usp=sharing), [officehome](https://drive.google.com/file/d/1hASLWAfkf4qj-WXU5cx9uw9rQDsDvSyO/view?usp=sharing). The splits need to be extracted to the two datasets' folders. Assume you put the datasets under the directory `$DATA`, the structure should look like
```
$DATA/
|–– pacs/
|   |–– images/
|   |–– splits/
|   |–– splits_ssdg/
|–– office_home_dg/
|   |–– art/
|   |–– clipart/
|   |–– product/
|   |–– real_world/
|   |–– splits_ssdg/
```

The style augmentation is based on [AdaIN](https://arxiv.org/abs/1703.06868) and the implementation is based on this code https://github.com/naoto0804/pytorch-AdaIN. Please download the weights of the decoder and the VGG from https://github.com/naoto0804/pytorch-AdaIN and put them under a new folder `ssdg-benchmark/weights`.

## How to run StyleMatch

The script is provided in `ssdg-benchmark/scripts/StyleMatch/run_ssdg.sh`. You need to update the `DATA` variable that points to the directory where you put the datasets. There are three input arguments: `DATASET`, `NLAB` (total number of labels), and `CFG`. See the tables below regarding how to set the values for these variables.

| `Dataset` | `NLAB` |
|---|---|
|`ssdg_pacs`| 210 or 105 |
|`ssdg_officehome`| 1950 or 975 |

|`CFG`| Description |
|---|---|
|`v1`| FixMatch + stochastic classifier + T_style |
|`v2`| FixMatch + stochastic classifier + T_style-only (i.e. no T_strong) |
|`v3`| FixMatch + stochastic classifier |
|`v4`| FixMatch |

`v1` refers to StyleMatch, which is our final model. See the config files in `configs/trainers/StyleMatch` for the detailed settings.

Here we give an example. Say you want to run StyleMatch on PACS under the 10-labels-per-class setting (i.e. 210 labels in total), simply run the following commands in your terminal,
```bash
conda activate dassl
cd ssdg-benchmark/scripts/StyleMatch
bash run_ssdg.sh ssdg_pacs 210 v1
```

In this case, the code will run StyleMatch in four different setups (four target domains), each for five times (five random seeds). You can modify the code to run a single experiment instead of all at once if you have multiple GPUs.

At the end of training, you will have
```
output/
|–– ssdg_pacs/
|   |–– nlab_210/
|   |   |–– StyleMatch/
|   |   |   |–– resnet18/
|   |   |   |   |–– v1/ # contains results on four target domains
|   |   |   |   |   |–– art_painting/ # contains five folders: seed1-5
|   |   |   |   |   |–– cartoon/
|   |   |   |   |   |–– photo/
|   |   |   |   |   |–– sketch/
```

To show the results, simply do
```bash
python parse_test_res.py output/ssdg_pacs/nlab_210/StyleMatch/resnet18/v1 --multi-exp
```

## Citation
If you use this code in your research, please cite our paper
```
@article{zhou2021stylematch,
    title={Semi-Supervised Domain Generalization with Stochastic StyleMatch},
    author={Zhou, Kaiyang and Loy, Chen Change and Liu, Ziwei},
    journal={arXiv preprint arXiv:2106.00592},
    year={2021}
}
```
