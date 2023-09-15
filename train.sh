CUDA='1'
N_GPU=1
BATCH=128
DATA=data/
IMAGENETS=data/

DUMP_PATH=./weights/pass50
DUMP_PATH_FINETUNE=${DUMP_PATH}/pixel_attention
DUMP_PATH_SEG=${DUMP_PATH}/pixel_finetuning
QUEUE_LENGTH=2048
QUEUE_LENGTH_PIXELATT=3840
HIDDEN_DIM=512
NUM_PROTOTYPE=500
ARCH=resnet18
NUM_CLASSES=50
EPOCH=2
EPOCH_PIXELATT=2
EPOCH_SEG=2
FREEZE_PROTOTYPES=1001
FREEZE_PROTOTYPES_PIXELATT=0

mkdir -p ${DUMP_PATH_FINETUNE}
mkdir -p ${DUMP_PATH_SEG}

# CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python main_pretrain.py \
CUDA_VISIBLE_DEVICES=${CUDA} python main_pretrain.py \
--arch ${ARCH} \
--data_path ${DATA}/train \
--dump_path ${DUMP_PATH} \
--nmb_crops 2 6 \
--size_crops 224 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--hidden_mlp ${HIDDEN_DIM} \
--nmb_prototypes ${NUM_PROTOTYPE} \
--queue_length ${QUEUE_LENGTH} \
--epoch_queue_starts 15 \
--epochs ${EPOCH} \
--batch_size ${BATCH} \
--base_lr 0.6 \
--final_lr 0.0006  \
--freeze_prototypes_niters ${FREEZE_PROTOTYPES} \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 0 \
--seed 31 \
--shallow 3 \
--weights 1 1

# CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python main_pixel_attention.py \
CUDA_VISIBLE_DEVICES=${CUDA} python main_pixel_attention.py \
--arch ${ARCH} \
--data_path ${IMAGENETS}/train \
--dump_path ${DUMP_PATH_FINETUNE} \
--nmb_crops 2 \
--size_crops 224 \
--min_scale_crops 0.08 \
--max_scale_crops 1. \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--hidden_mlp ${HIDDEN_DIM} \
--nmb_prototypes ${NUM_PROTOTYPE} \
--queue_length ${QUEUE_LENGTH_PIXELATT} \
--epoch_queue_starts 0 \
--epochs ${EPOCH_PIXELATT} \
--batch_size ${BATCH} \
--base_lr 6.0 \
--final_lr 0.0006  \
--freeze_prototypes_niters ${FREEZE_PROTOTYPES_PIXELATT} \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 0 \
--seed 31 \
--pretrained ${DUMP_PATH}/checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=${CUDA} python cluster.py -a ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE} \
-c ${NUM_CLASSES}

### Evaluating the pseudo labels on the validation set.
CUDA_VISIBLE_DEVICES=${CUDA} python inference_pixel_attention.py -a ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE} \
-c ${NUM_CLASSES} \
--mode validation \
--test \
--centroid ${DUMP_PATH_FINETUNE}/cluster/centroids.npy

# CUDA_VISIBLE_DEVICES=${CUDA} python evaluator.py \
# --predict_path ${DUMP_PATH_FINETUNE} \
# --data_path ${IMAGENETS} \
# -c ${NUM_CLASSES} \
# --mode validation \
# --curve \
# --min 20 \
# --max 80

# CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python inference_pixel_attention.py -a ${ARCH} \
CUDA_VISIBLE_DEVICES=${CUDA} python inference_pixel_attention.py -a ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE} \
-c ${NUM_CLASSES} \
--mode train \
--centroid ${DUMP_PATH_FINETUNE}/cluster/centroids.npy \
-t 0.28

# CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python main_pixel_finetuning.py \
CUDA_VISIBLE_DEVICES=${CUDA} python main_pixel_finetuning.py \
--arch ${ARCH} \
--data_path ${DATA}/train \
--dump_path ${DUMP_PATH_SEG} \
--epochs ${EPOCH_SEG} \
--batch_size ${BATCH} \
--base_lr 0.6 \
--final_lr 0.0006 \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 0 \
--num_classes ${NUM_CLASSES} \
--pseudo_path ${DUMP_PATH_FINETUNE}/train \
--pretrained ${DUMP_PATH}/checkpoint.pth.tar

#######################################
# 以上的内容为模型的训练部分，可以直接替换成现有的pass的模型 *end-of-train*
#######################################
MODEL_1="pass50"

# 生成match文件（用于多模型融合时伪标签的映射）
CUDA_VISIBLE_DEVICES=${CUDA} python inference.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG}/checkpoint.pth.tar \
--model1_name ${MODEL_1} \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG} \
-c ${NUM_CLASSES} \
--mode match_generate \

# 利用训练好的pass模型推理 训练集 得到预测结果
CUDA_VISIBLE_DEVICES=${CUDA} python inference.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG}/checkpoint.pth.tar \
--model1_name ${MODEL_1} \
--model_tta_method 'flip|blur' \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG} \
-c ${NUM_CLASSES} \
--mode train \
--is_test_b 0 \
--match_file ${DUMP_PATH_SEG}/validation/match_${MODEL_1}.json
#--pretrained 指定第一个模型的路径
#--model1_name 指定第一个模型的名字，用于寻找对应的match.json
#--pretrained_extra 指定第二个模型的名字，第二个模型不存在时不必填写，此时跟模型融合相关的参数都不会生效
#--model2_name 指定第二个模型的名字
#--model_merge_method 指定模型融合的方法，有取平均avg与取最大值max两个可选项
#--model_tta_method 指定测试时数据增强方法，需要的参数用|分隔，可选项有flip，blur，gamma，brightness

# 生成sam的运行结果
# 使用前先将sam所使用的torch格式的vit-b的模型转换成jittor所需的格式
# 有一些原图尺寸很大，如果处理不了在amg.py中取消用于降低分辨率的代码的注释即可
CUDA_VISIBLE_DEVICES=0 python amg.py \
--checkpoint ./weights/sam/sam_vit_b_01ec64.pth.tar \
--model-type vit_b \
--input ${IMAGENETS}/train \
--output ./weights/sam/output/train \


# 合并网络推理的结果和sam推理的结果，生成新的伪标签
BATCH=64
DUMP_PATH_SEG=${DUMP_PATH}/pixel_finetuning/unet
MODEL_1="unet"

mkdir -p ${DUMP_PATH_SEG}

CUDA_VISIBLE_DEVICES=${CUDA} python integrate_sam_inference.py \
--input data/ImageNetS50/ \
--points_count 12 \
--distribution normal \
--checkpoint ./weights/sam/sam_vit_b_01ec64.pth.tar \
--model_type vit_b \
--inference_result ./weights/pass50/pixel_finetuning/validation/"CHANGE THIS TO YOUR GENERATE FOLDER PATH"  \
--sam_result ./weights/sam/output \
--output ${DUMP_PATH_SEG}/merged_label \
--mode train


# 利用u-shape的resnet再次对模型进行一次微调
CUDA_VISIBLE_DEVICES=${CUDA} python main_pixel_finetuning.py \
--arch ${ARCH} \
--data_path ${DATA}/train \
--dump_path ${DUMP_PATH_SEG} \
--epochs ${EPOCH_SEG} \
--batch_size ${BATCH} \
--base_lr 0.6 \
--final_lr 0.0006 \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 0 \
--apply_unet \
--num_classes ${NUM_CLASSES} \
--pseudo_path ${DUMP_PATH_SEG}/merged_label \
--pretrained ./weights/pass50/pixel_finetuning/checkpoint.pth.tar


# #######################################
# # 推理验证集与评估
# #######################################

# 生成match文件（主要用于多模型融合时伪标签的映射，不可跳过）
CUDA_VISIBLE_DEVICES=${CUDA} python inference.py -a ${ARCH} \
--pretrained ./weights/pass50/pixel_finetuning/unet/checkpoint.pth.tar \
--model1_name ${MODEL_1} \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG} \
-c ${NUM_CLASSES} \
--mode match_generate \

# 生成验证集结果
CUDA_VISIBLE_DEVICES=${CUDA} python inference.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG}/checkpoint.pth.tar \
--model1_name ${MODEL_1} \
--model_tta_method 'flip|blur' \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG} \
-c ${NUM_CLASSES} \
--mode validation \
--is_test_b 0 \
--match_file ${DUMP_PATH_SEG}/validation/match_${MODEL_1}.json

# 评估验证集结果
CUDA_VISIBLE_DEVICES=${CUDA} python evaluator.py \
--predict_path ${DUMP_PATH_SEG}/validation/"CHANGE THIS TO YOUR GENERATE FOLDER PATH" \
--data_path ${IMAGENETS} \
-c ${NUM_CLASSES} \
--mode validation

# 生成测试集结果
CUDA_VISIBLE_DEVICES=${CUDA} python inference.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG}/checkpoint.pth.tar \
--model1_name ${MODEL_1} \
--model_tta_method 'flip|blur' \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG} \
-c ${NUM_CLASSES} \
--mode test \
--is_test_b 1 \
--match_file ${DUMP_PATH_SEG}/validation/match_${MODEL_1}.json