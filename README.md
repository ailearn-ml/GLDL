# Code for DICE-VI and DICE-CC

Code for "Delving into Generalizable Label Distribution Learning" in TPAMI 2025.

If you use the code in this repo for your work, please cite the following BibTeX entries:

```
@article{GLDL,
	author    = {Xingyu Zhao and
	Lei Qi and
	Yuexuan An and
	Xin Geng},
	title     = {Delving into Generalizable Label Distribution Learning},
	journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
	year      = {2025},
}
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.13
- NumPy >= 1.13.3
- Scikit-learn >= 0.20

## Getting started

The original datasets of the Visual Sentiment Distribution (VSD), the Facial Beauty Perception (FBP), and the Movie Rating Distribution (MRD) benchmark datasets curated in this paper are from [Flickr\_LDL](https://ojs.aaai.org/index.php/AAAI/article/view/10485), [Twitter\_LDL](https://ojs.aaai.org/index.php/AAAI/article/view/10485), [Abstract Painting](https://dl.acm.org/doi/10.1145/1873951.1873965), [FBP5500](https://ieeexplore.ieee.org/document/8546038/), and [the movies dataset](https://ieeexplore.ieee.org/document/8546038/).  We provide the extracted features for these benchmark datasets [here](https://drive.google.com/file/d/1LDuJiv3K1KVGYyLUeSDGT2t3eh9QKZ0G/view?usp=drive_link). 

Unzip the download file and put it into the main directory.

## Running the scripts

To train and test the DICE-VI model in the terminal, use:

```bash
$ python run_dice_vi.py --dataset VSD --lambda1 0.1 --lambda2 0.1 --max_epoch 50 --batch_size 32 --lr 0.001 --device cuda:0 --seed 0
```


To train and test the DICE-CC model in the terminal, use:

```bash
$ python run_dice_cc.py --dataset VSD --loss_type focal --alpha 0.01 --max_epoch 50 --batch_size 32 --lr 0.001 --device cuda:0 --seed 0
```

## Acknowledgment

Our project references the datasets in the following repositories and papers.

[Learning Visual Sentiment Distributions via Augmented Conditional Probability Neural Network. AAAI 2017: 224-230.](https://ojs.aaai.org/index.php/AAAI/article/view/10485)

[Affective Image Classification Using Features Inspired by Psychology and Art Theory. ACM Multimedia 2010: 83-92.](https://dl.acm.org/doi/10.1145/1873951.1873965)

[SCUT-FBP5500: A Diverse Benchmark Dataset for Multi-Paradigm Facial Beauty Prediction. ICPR 2018: 1598-1603.](https://ieeexplore.ieee.org/document/8546038/)

[Movie metadata](https://www.kaggle.com/datasets/karrrimba/movie-metadatacsv/data)





