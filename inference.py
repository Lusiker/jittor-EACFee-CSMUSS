import argparse
import json
import os
from datetime import datetime
import shutil
import albumentations as A

import numpy as np
import jittor as jt
import jittor.nn as nn
jt.flags.use_cuda = 1
from PIL import Image
import jittor.transform as transforms
from tqdm import tqdm

import src.resnet as resnet_model
from src.singlecropdataset import InferImageFolder
from src.utils import hungarian

import pdb #


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--dump_path',
                        type=str,
                        default=None,
                        help='The path to save results.')
    parser.add_argument('--match_file',
                        type=str,
                        default=None,
                        help='The matching file for test set.')
    parser.add_argument('--data_path',
                        type=str,
                        default=None,
                        help='The path to ImagenetS dataset.')
    parser.add_argument('--pretrained',
                        type=str,
                        default=None,
                        help='The model checkpoint file.')
    parser.add_argument('--pretrained_extra',
                        type=str,
                        default=None,
                        help='The extra model checkpoint file.'
    )
    parser.add_argument('--model1_name',
                        type=str,
                        default='model1',
                        help='Name of the first model'
    )
    parser.add_argument('--model2_name',
                        type=str,
                        default='model2',
                        help='Name of the second model'
    )
    parser.add_argument('--model_merge_method',
                        type=str,
                        default='null',
                        help='method of model merging'
    )
    parser.add_argument('--model_tta_method',
                        type=str,
                        default='null',
                        help='methods of model tta, arguments divided by |'
    )
    parser.add_argument('-a',
                        '--arch',
                        metavar='ARCH',
                        help='The model architecture.')
    parser.add_argument('-c',
                        '--num-classes',
                        default=50,
                        type=int,
                        help='The number of classes.')
    parser.add_argument('--max_res',
                        default=1000,
                        type=int,
                        help="Maximum resolution for evaluation. 0 for disable.")
    parser.add_argument('--method',
                        default='example submission',
                        help='Method name in method description file(.txt).')
    parser.add_argument('--train_data',
                        default='null',
                        help='Training data in method description file(.txt).')
    parser.add_argument('--train_scheme',
                        default='null',
                        help='Training scheme in method description file(.txt), \
                        e.g., SSL, Sup, SSL+Sup.')
    parser.add_argument('--link',
                        default='null',
                        help='Paper/project link in method description file(.txt).')
    parser.add_argument('--description',
                        default='null',
                        help='Method description in method description file(.txt).')
    parser.add_argument('--single_pic',
                        default='null',
                        help='when not null, worker only processes one picture'
    )
    parser.add_argument('--is_test_b',
                        type=int,
                        default=False,
                        help='whether current inference target is test B'
    )
    args = parser.parse_args()

    return args


# i(match1-fake) -> real -> match2-fake
def match_psuedo_label(i: int, match1: dict[str, int], match2: dict[str, int]):
    real_label = match1[str(i)]
    for k, v in match2.items():
        if v == real_label:
            return int(k)


def merge(output1, output2, match1: dict[str, int], match2: dict[str, int], method: str):
    if method == 'max':
        output_temp = jt.zeros([
            output1.shape[0], 
            output1.shape[1] * 2,
            output1.shape[2], 
            output1.shape[3], 
        ])

        output_temp[:, :output1.shape[1], :, :] = output1[:, :, :, :]
        for i in range(output1.shape[1]):
            output_temp[:, output1.shape[1] + i, :, :] = output2[:, match_psuedo_label(i, match1, match2), :, :]

        return output_temp
    
    elif method == 'avg':
        output_temp = jt.zeros([
            output1.shape[0], 
            output1.shape[1],
            output1.shape[2], 
            output1.shape[3], 
        ])

        output_temp[:, :, :, :] = output1[:, :, :, :]
        for i in range(output1.shape[1]):
            output_temp[:, i, :, :] += output2[:, match_psuedo_label(i, match1, match2), :, :]

        return output_temp
    else:
        pass


class TTAProcessor:
    def __init__(self, methods):
        self.method_dict = {}
        for method in methods:
            if method == 'flip':
                self.method_dict[method] = A.Flip(p=1.0)
            elif method == 'blur':
                self.method_dict[method] = A.Blur(p=1.0)
            elif method == 'brightness':
                self.method_dict[method] = A.RandomBrightnessContrast(p=1.0)
            elif method == 'gamma':
                self.method_dict[method] = A.RandomGamma(p=1.0)
            else:
                pass # unknown tta method

    def process(self, image, method):
        return self.method_dict[method].apply(image)


def tta(image, model, methods, processor):
    if len(methods) == 0:
        return model(image)

    results = []
    for i in range(image.shape[0]):
        # 全部tta方法的结果存至此处
        current_inputs = jt.zeros([
            len(methods) + 1, 
            image.shape[1],
            image.shape[2], 
            image.shape[3], 
        ])

        # 先把原图存到0
        original_image = image[i, :, :, :]
        current_inputs[0, :, :, :] = original_image
        
        # 按指定方法分别获得变换结果,存到current_inputs的后面
        for index in range(len(methods)):
            processed = processor.process(original_image.numpy(), methods[index])
            current_inputs[index + 1, :, :, :] = processed

        # 将由一张图得到的全部结果输入模型，得到预测
        output_temp = model(current_inputs)
        
        results.append(output_temp)
    
    result = jt.zeros([
        image.shape[0],
        results[0].shape[0] * results[0].shape[1],
        results[0].shape[2],
        results[0].shape[3]
    ])

    for i in range(len(results)):
        for j in range(len(methods) + 1):
            result[
                i, 
                j * results[0].shape[1] : (j + 1) * results[0].shape[1],
                :, :
            ] = results[i][j, :, :]
    
    return result


def merge_tta_output(output, methods):
    if len(methods) == 0:
        return output

    c = output.shape[1] // (len(methods) + 1)
    result = jt.zeros([
        output.shape[0],
        c,
        output.shape[2],
        output.shape[3]
    ])


    for i in range(len(methods)):
        current_output = output[:, (i + 1) * c : (i + 2) * c, :, :]
        if methods[i] == 'flip':
            current_output = current_output[:, :, jt.arange(output.shape[2] - 1, -1, -1), :]

        result[:, 0 : c, :, :] += current_output[:, :, :, :]
    
    for i in range(output.shape[0]):
        #outputs = output[i, :, :, :]
        # outputs = outputs.reshape((len(methods) + 1), c, output.shape[2], output.shape[3])
        # outputs = jt.argmax(outputs, dim=0, keepdims=True)[1]

        result[i, :, :, :] = output[i, 0 : c, :, :]

    return result


def main_worker(args):
    is_single_mode = args.single_pic != 'null'
    is_test_b = args.is_test_b == 1
    
    # build model
    if 'resnet' in args.arch:
        model = resnet_model.__dict__[args.arch](
            hidden_mlp=0, output_dim=0, nmb_prototypes=0, train_mode='finetune', num_classes=args.num_classes, apply_unet='unet' in args.model1_name)
        model1_name = args.model1_name

        if args.pretrained_extra != None:
            # 当bash脚本指定了第二个模型，为其创建一个ResNet实例
            model_extra = resnet_model.__dict__[args.arch](
                hidden_mlp=0, output_dim=0, nmb_prototypes=0, train_mode='finetune', num_classes=args.num_classes, apply_unet=args.model2_name == 'unet')    
            model2_name = args.model2_name
        else:
            model_extra = None # 参数未给出，第二个模型为None
            model2_name = 'null'
    
    else:
        raise NotImplementedError()
    
    # 读取第一个模型
    try:
        checkpoint = jt.load(args.pretrained)["state_dict"]
    except FileNotFoundError:
        # 第一个模型路径错误，退出执行
        print(f'Wrong file path for model:{args.pretrained}')
        exit()
    for k in list(checkpoint.keys()):
        if k not in model.state_dict().keys():
            del checkpoint[k]
        model.load_state_dict(checkpoint)
    print("=> loaded model '{}'".format(args.pretrained))

    # 读取第二个模型（如果存在）
    if model_extra != None:
        try:
            checkpoint_extra = jt.load(args.pretrained_extra)["state_dict"]
        except FileNotFoundError:
            # 第二个模型路径错误，退出执行
            print(f'Wrong file path for extra model:{args.pretrained_extra}')
            exit()
        
        #将参数读入额外模型
        for k in list(checkpoint_extra.keys()):
            if k not in model_extra.state_dict().keys():
                del checkpoint[k]
        model_extra.load_state_dict(checkpoint_extra)
        print("=> loaded EXTRA model '{}'".format(args.pretrained_extra))

    if args.mode != 'match_generate':
        inference_start_time = datetime.now()

        # 标记本次推理的详细信息，将以此生成路径
        identifier = '-'.join((
            model1_name + (f'_{model2_name}' if model_extra != None else ''), #模型名称
            f'{args.model_merge_method}_merge', # 使用的融合方法
            f'{args.model_tta_method.replace("|", "_")}_tta', #使用的tta方法
            inference_start_time.strftime("%Y-%m-%d_%H-%M-%S"), #推理开始的时间
            args.mode #当前模式
        ))

        tta_methods = [] if args.model_tta_method == 'null' else [method for method in args.model_tta_method.split('|')]
        tta_processor = TTAProcessor(tta_methods)

    model.eval()
    if model_extra != None:
        model_extra.eval()

    # build dataset
    assert args.mode in ('match_generate', 'train', 'validation', 'test')
    assert args.model_merge_method in ('max', 'avg', 'null')

    if args.mode != 'test':
        data_path = os.path.join(args.data_path, 'train' if args.mode == 'train' else 'validation')
    else:
        if is_test_b:
            data_path = os.path.join(args.data_path, 'testB')
        else:
            data_path = os.path.join(args.data_path, 'test')

    validation_segmentation = os.path.join(args.data_path,
                                           'validation-segmentation')
    normalize = transforms.ImageNormalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = InferImageFolder(root=data_path,
                               transform=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.ToTensor(),
                                   normalize,
                               ]))
    dataloader = dataset.set_attrs(
        batch_size=1, 
        num_workers=1
    )

    dump_path = os.path.join(args.dump_path, 'validation' if args.mode != 'test' else 'test')
    load_path = os.path.join(args.dump_path, 'validation')

    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    targets = [] # 按顺序得到的真实类号
    predictions = [] # 按顺序得到的单张图片的单个类号
    predictions_extra = [] # 第二个模型按顺序得到的单张图片的单个类号
    skip_count = 14 # 跳过图片数

    # 非匹配生成模式时，先读取已有的匹配文件，并构建字典
    if args.mode != 'match_generate':
        with open(os.path.join(load_path, f'match_{model1_name}.json'), 'r') as match1_raw:
            match1 = json.loads(match1_raw.read())
        
        # # NOTE:
        # with open("./weights/res50/pixel_finetuning/test/match.json", 'r') as match2_raw:
        #     match2 = json.loads(match2_raw.read())
        
        if model_extra != None:
            with open(os.path.join(load_path, f'match_{model2_name}.json'), 'r') as match2_raw:
                match2 = json.loads(match2_raw.read())         

    for images, path, height, width in tqdm(dataloader):
        if is_single_mode and skip_count > 0:
            # 参数中指定单张图片模式下按skip_count跳过图片
            skip_count -= 1
            continue

        path = path[0]
        cate = path.split('/')[-2]
        name = path.split('/')[-1].split('.')[0]
        
        # if not os.path.exists(os.path.join(dump_path, cate)):
        #     os.makedirs(os.path.join(dump_path, cate))

        with jt.no_grad():
            H = height.item()
            W = width.item()

            if args.mode == 'match_generate':
                # 生成匹配模式

                output = model(images) #256 * 256 * 3 ==model==> 51 * 8 * 8
                if model_extra != None:
                    output_extra = model_extra(images)

                if H * W > args.max_res * args.max_res and args.max_res > 0:
                    output1 = nn.interpolate(output, (args.max_res, int(args.max_res * W / H)), mode="bilinear", align_corners=False)
                    if model_extra != None:
                        output2 = nn.interpolate(output_extra, (args.max_res, int(args.max_res * W / H)), mode="bilinear", align_corners=False)
                    
                    # jt.argmax --> 最大值下标, 最大值
                    # output取最大值下标,取dim=1意味着对51层H*W中找到最大的层号
                    output1 = jt.argmax(output1, dim=1, keepdims=True)[0]
                    if model_extra != None:
                        output2 = jt.argmax(output2, dim=1, keepdims=True)[0]

                    # 通过插值变回原大小
                    prediction1 = nn.interpolate(output1.float(), (H, W), mode="nearest").long()
                    if model_extra != None:
                        prediction2 = nn.interpolate(output2.float(), (H, W), mode="nearest").long()
                else:
                    output1 = nn.interpolate(output, (H, W), mode="bilinear", align_corners=False)
                    if model_extra != None:
                        output2 = nn.interpolate(output_extra, (H, W), mode="bilinear", align_corners=False)

                    prediction1 = jt.argmax(output1, dim=1, keepdims=True)[0]
                    if model_extra != None:
                        prediction2 = jt.argmax(output2, dim=1, keepdims=True)[0]

                target = Image.open(os.path.join(validation_segmentation, cate, name + '.png'))
                target = np.array(target).astype(np.int32)
                target = target[:, :, 1] * 256 + target[:, :, 0]

                # Prepare for matching (target)
                target_unique = np.unique(target.reshape(-1))
                target_unique = target_unique - 1
                target_unique = target_unique.tolist()
                if -1 in target_unique:
                    target_unique.remove(-1)
                targets.append(target_unique)

                # Prepare for matching (prediction)
                prediction_unique = np.unique(prediction1.cpu().numpy().reshape(-1))
                prediction_unique = prediction_unique - 1
                prediction_unique = prediction_unique.tolist()
                if -1 in prediction_unique:
                    prediction_unique.remove(-1)
                predictions.append(prediction_unique)

                # Prepare for matching (prediction_extra)
                if model_extra != None:
                    prediction_extra_unique = np.unique(prediction2.cpu().numpy().reshape(-1))
                    prediction_extra_unique = prediction_extra_unique - 1
                    prediction_extra_unique = prediction_extra_unique.tolist()
                    if -1 in prediction_extra_unique:
                        prediction_extra_unique.remove(-1)
                    predictions_extra.append(prediction_extra_unique)

            else:
                # 推理
                output = tta(images, model, tta_methods, tta_processor)
                if model_extra != None:
                    output_extra = tta(images, model_extra, tta_methods, tta_processor)

                if H * W > args.max_res * args.max_res and args.max_res > 0:
                    # 原图过大的情况
                    if model_extra != None:
                        # 两个模型
                        output1 = nn.interpolate(output, (args.max_res, int(args.max_res * W / H)), mode="bilinear", align_corners=False)
                        output2 = nn.interpolate(output_extra, (args.max_res, int(args.max_res * W / H)), mode="bilinear", align_corners=False)
                        output1 = merge_tta_output(output1, tta_methods)
                        output2 = merge_tta_output(output2, tta_methods)

                        output = merge(output1, output2, match1, match2, args.model_merge_method)

                        # jt.argmax --> 最大值下标, 最大值
                        # output取最大值下标,取dim=1意味着对51层H*W中找到最大的层号
                        output = jt.argmax(output, dim=1, keepdims=True)[0]

                        # 通过插值变回原大小
                        prediction = nn.interpolate(output.float(), (H, W), mode="nearest").long()
                        prediction = prediction % output1.shape[1]

                    else:
                        # 单个模型
                        output = nn.interpolate(output, (args.max_res, int(args.max_res * W / H)), mode="bilinear", align_corners=False)
                        output = merge_tta_output(output, tta_methods)
                        output = jt.argmax(output, dim=1, keepdims=True)[0]
                        prediction = nn.interpolate(output.float(), (H, W), mode="nearest").long()

                else:
                    if model_extra != None:
                        # 两个模型
                        output1 = nn.interpolate(output, (H, W), mode="bilinear", align_corners=False)
                        output2 = nn.interpolate(output_extra, (H, W), mode="bilinear", align_corners=False)
                        output1 = merge_tta_output(output1, tta_methods)
                        output2 = merge_tta_output(output2, tta_methods)

                        output = merge(output1, output2, match1, match2, args.model_merge_method)

                        prediction = jt.argmax(output, dim=1, keepdims=True)[0]
                        prediction = prediction % output1.shape[1]

                    else:
                        # 单个模型
                        output = nn.interpolate(output, (H, W), mode="bilinear", align_corners=False)
                        output = merge_tta_output(output, tta_methods)
                        prediction = jt.argmax(output, dim=1, keepdims=True)[0]

                prediction = prediction.squeeze(0).squeeze(0)

                if not os.path.exists(os.path.join(dump_path, identifier, cate)):
                    os.makedirs(os.path.join(dump_path, identifier, cate))

                res = jt.zeros((prediction.shape[0], prediction.shape[1], 3))
                res[:, :, 0] = prediction % 256
                res[:, :, 1] = prediction // 256
                res = res.cpu().numpy()

                ## NOTE: 添加对空预测图的处理，若预测结果为空则去另一个模型的结果里捞
                # (这样做应该不算是trick而算是加入额外监督了，网络是不知道哪个模型有结果的)
                # 应该也不算作弊，算多模型融合
                # if not np.any(res):
                #     another_result_path = os.path.join("./weights/res50/pixel_finetuning/test", cate, name + '.png')
                #     another_pred = Image.open(another_result_path).convert('RGB')
                #     another_pred = np.array(another_pred)

                #     for i in range(51):
                #         res[another_pred == match_psuedo_label(i, match1, match2)] = i

                res = Image.fromarray(res.astype(np.uint8))

                res_path = os.path.join(dump_path, identifier, cate, name + '.png') #
                res.save(res_path)
            
            jt.clean_graph()
            jt.sync_all()
            jt.gc()

        if is_single_mode:
            break
    
    if args.mode == 'match_generate':
        #匹配表的键值对为 伪标签:真实标签

        # 构建并保存匹配表1
        _, match1 = hungarian(targets, predictions, num_classes=args.num_classes)
        match1 = {k + 1: v + 1 for k, v in match1.items()}
        match1[0] = 0

        with open(os.path.join(dump_path, f'match_{model1_name}.json'), 'w') as f:
            f.write(json.dumps(match1))

        if model_extra != None:
            _, match2 = hungarian(targets, predictions_extra, num_classes=args.num_classes)
            match2 = {k + 1: v + 1 for k, v in match2.items()}
            match2[0] = 0
            
            # 构建并保存匹配表2
            with open(os.path.join(dump_path, f'match_{model2_name}.json'), 'w') as f:
                f.write(json.dumps(match2))

    elif args.mode == 'validation':
        pass

    elif args.mode == 'test':
        assert os.path.exists(args.match_file)
        # zip for submission

        target_path = os.path.join(args.dump_path, args.mode, identifier)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        shutil.copyfile(args.match_file, os.path.join(target_path, 'match.json'))

        method = 'Method name: {}\n'.format(args.method) + \
            'Training data: {}\nTraining scheme: {}\n'.format(
                args.train_data, args.train_scheme) + \
            'Networks: {}\nPaper/Project link: {}\n'.format(
                args.arch, args.link) + \
            'Method description: {}'.format(args.description)
        with open(os.path.join(target_path, 'method.txt'), 'w') as f:
            f.write(method)

        shutil.make_archive(target_path, 'zip', root_dir=target_path)
        print(f'test zip generated at {target_path}')


if __name__ == '__main__':
    args = parse_args()
    main_worker(args=args)
