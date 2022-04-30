# HIR
This repo is the offical implementation for "Human Interaction Recognition with Skeletal Attention and Shift Graph Convolution". The paper is accepted to IJCNN2022. 
Note: We also provide a simple model, which achieves 100%, 90.63%  on UT  and BIT-Interaction , respectively.
# Prerequisite

 - PyTorch 1.5.0
 - Cuda 10.2
 - gcc 5.5.0
# Data Preparation
 - Download the raw data of UT and BIT-Interaction. 
 - Download the raw data for Campus-Interaction (CI) here: [Baidu,code:smd3](https://pan.baidu.com/s/1gDiufsvR-v7IdNCShXoYBg 
).
 - Use [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) to obtain skeletons and skeleton bboxes.
 - Use [enter link description here](https://github.com/ifzhang/FairMOT) to obtain tracking bboxes.
 - Combine skeleton bboxes and tracking bboxes with python `python compute_bbox.py`
 - Data preprocessing files to `./data/dataset_name`. One example can be downloaded [here](https://pan.baidu.com/s/1gDiufsvR-v7IdNCShXoYBg)
# Trianing & Testing
 - For UT
 `python main.py --dataset UT --stream appearance`
 `python main.py --dataset UT --stream pose`
 - For BIT
 `python main.py --dataset BIT --stream appearance`
 `python main.py --dataset BIT --stream pose`
  - For CI
 `python main.py --dataset CI --stream appearance`
 `python main.py --dataset CI --stream pose`
 # Two-stream ensemble
 To ensemble the results of two stream. Run `python ensemble.py`
 # Citation
    @inproceedings{Jin2022hir,
	    title = {Human Interaction Recognition with Skeletal Attention and Shift Graph Convolution},
	    author = {Jin Zhou and Zhenhua Wang and Jiajun Meng and Shen Liu and Jianhua Zhang and Shengyong chen},
	    booktitle = {2021 International Joint Conference on Neural Networks (IJCNN)},
	    year = {2022}
    }
