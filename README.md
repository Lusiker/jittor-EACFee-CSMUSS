# Jittor 大规模无监督语义分割赛道-

## 简介
本项目包含了第三届计图人工智能挑战赛赛道二-大规模无监督语义分割(LUSS)赛题的代码实现。本项目的主要特点如下：
1. 采用了PASS方法作为basic model。
2. 引入了基于SAM-ViT-B的类别-语义聚合模块增强伪标签质量。
3. 引入U-Shape模块改进网络分割头，增强网络的表达能力。

## 安装 
本项目可在1张4090上运行，训练时间约为48小时。

## 运行环境
- ubuntu 20.04 LTS
- python >= 3.8
- jittor >= 1.3.0

### 安装依赖
执行以下命令安装 python 依赖
```
pip install -r requirements.txt
```

### 数据准备


### 预训练模型
本项目需要使用预训练的SAM-ViT-B模型，下载地址为https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth 。随后，你需要在`./train.sh` 中运行amg.py的部分修改--checkpoint参数为SAM模型对应的地址。运行此段脚本能得到所有制定路径下所有图片的对应SAM分割mask。

## 训练
训练时，通过以下脚本中*end-of-train*标记之前的部分可以得到一个模型。
```
bash ./train.sh
```

## 推理、验证与测试

得到模型后，你可以通过运行train.sh中对应部分完成推理、验证与生成测试结果。以下是一些重要的细节：
1. 对于得到的模型*model*，你首先应该指定--mode参数为*match_generate*，从中得到一个以*model*为名的match.json文件。
2. 获得match.json后，你可以修改--mode参数为validation或test来进行对应数据的推理，选择复赛的test b数据时应额外指定--is_test_b参数为1。
若你有两个待融合的模型，你可以分别指定--pretrained，--pretrained_extra，--model1_name，--model2_name,--model_merge_method这五个参数进行模型融合（只有一个模型的情况下只需要指定--pretrained与--model1_name。你还可以通过指定--model_tta_method参数来进行选择测试时数据增强，详细细节见train.sh中的注释。
3. 得到推理结果后，你可以运行integrate_sam_inference.py对应的脚本获得经由类别-语义聚合模块增强的伪标签结果。此处，你需要通过--inference_result指定上一步生成的伪标签结果。注意，你必须确保已经得到数据预处理部分得到的SAM分割结果，并且确保--mode参数与输入数据的模式匹配（可选值为train、validation、test和testB）。此步的结果会存放于--output参数指定的位置。
4. 对于inference.py部分的推理任意推理结果（不论是否经过了integrate_sam_inference的处理），如果是验证集，你可以运行evaluator.py对应的部分得到验证集上的表现评估结果；如果是测试集，你可以找到并将脚本打包的结果上传评估网站。

## 微调
对于模型新生成的伪标签数据，你可以通过运行main_pixel_finetuning.py对应的部分，通过--pseudo_path指定新伪标签的路径来进行模型的微调。若你需要使用U-Shape上采样模块，那么你需要显式添加--apply_unet参数（若模型已经经历过U-Shape微调阶段，则必须添加此参数，否则会引发错误）。

## 致谢
此项目基于论文 [Large-scale Unsupervised Semantic Segmentation](https://arxiv.org/abs/2106.03149) 实现。
