import argparse
import os
from datetime import datetime
from PIL import Image

from segment_anything import SamPredictor, sam_model_registry

import cv2
import numpy as np
import jittor as jt
jt.flags.use_cuda = 1

from tqdm import tqdm


parser = argparse.ArgumentParser(
    description='Processes the inference result by integrating generated pseudo label pictures with segment anything masks'
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument(
    "--points_count",
    type=int,
    required=True,
    default=10,
    help="The count of points chosen from pseudo label tensor, which will be passed to sam as prompt",
)

parser.add_argument(
    "--distribution",
    type=str,
    required=True,
    default='normal',
    help="The distribution of points' coordinate",
)

parser.add_argument(
    "--model_type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--inference_result",
    type=str,
    required=True,
    help="root of pictures from a previous inference result",
)

parser.add_argument(
    "--sam_result",
    type=str,
    required=True,
    help="root of pictures from a previous sam result",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="saving root of integration results",
)

parser.add_argument(
    '--mode',
    type=str,
    help='target mode of pictures'
)


class InferenceResultLoader(jt.dataset.ImageFolder):
    def __init__(self, root, num_gpus=1):
        super().__init__(root, transform=None)

        if len(self.imgs) % num_gpus != 0:
            padding = num_gpus - len(self.imgs) % num_gpus
            for i in range(padding):
                self.imgs.append(self.imgs[i])
            self.total_len = len(self.imgs)
        
    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = Image.open(path).convert('RGB')
        height, width = img.size[1], img.size[0]

        return img, path, height, width


def get_indices_from_distribution(total, distribution, count):
    if distribution == 'normal':
        return jt.Var(np.random.choice(total, size=count, replace=False))
    else:
        raise NotImplementedError('Distribution not implemented yet:', distribution)


# def get_rc(img_path):
#     img3i = cv2.imread(img_path)
#     img3f = img3i.astype(np.float32)
#     img3f *= 1. / 255
    
#     sal = SaliencyRC.GetHC(img3f)

#     idxs = np.where(sal < (sal.max()+sal.min()) / 1.8)
#     sal[idxs] = 0
#     sal = sal * 255
#     sal = sal.astype(np.int16)

#     ret , sal = cv2.threshold(sal, 0, 255, cv2.THRESH_BINARY)

#     return sal


def remove_small_areas(tensor, n):
    num_labels, labeled_tensor = cv2.connectedComponents(tensor.astype(np.uint8))

    areas = np.bincount(labeled_tensor.flatten())[1:]

    small_area_indices = np.where(areas <= n)[0]
    for idx in small_area_indices:
        tensor[labeled_tensor == idx + 1] = 0

    return tensor


def main(args):
    assert(args.mode in ('train', 'test', 'validation', 'testB'))

    print(f"Loading SAM model...@{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    mask_predictor = SamPredictor(sam)
    print(f"SAM model loaded@{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    dataset = InferenceResultLoader(args.inference_result)
    dataloader = dataset.set_attrs(
        batch_size=1, 
        num_workers=8
    )

    processed_log = []
    for image, path, H, W in tqdm(dataloader):
        image = image[0]

        path = path[0]
        class_id = path.split('/')[-2]
        pic_name = (path.split('/')[-1]).split('.')[0]

        as_pseudo = jt.zeros([image.shape[0], image.shape[1], 1], dtype=jt.int32)
        as_pseudo[:, : ,0] += jt.argmax(image, dim=2, keepdims=False)[1]

        
        labels, counts = np.unique(as_pseudo.reshape(-1), return_counts=True)
        labels, counts = labels.tolist(), counts.tolist()

        # 移除所有空标签及其数量
        try:
            zero_index = labels.index(0)
            labels.pop(zero_index)
            counts.pop(zero_index)

            max_label = labels[counts.index(max(counts))]
            if len(labels) > 1:
                # 存在多个非零标签时，把少的全部换成多的
                as_pseudo = jt.where(as_pseudo != 0, max_label, 0)
        except ValueError:
            # 空标签不存在时不做处理
            pass
        
        if len(labels) == 0:
            # 当前推测无标签，直接保存旧结果
            res = Image.fromarray(image.cpu().numpy().astype(np.uint8))
            res_path = os.path.join(args.output, class_id)
            if not os.path.exists(res_path):
                os.makedirs(res_path)
            res.save(os.path.join(res_path, pic_name + '.png'))
            now = datetime.now()
            processed_log.append(f"{len(processed_log)} {now.strftime('%Y-%m-%d_%H-%M-%S')},{class_id}-{pic_name} no prediction, saved as original state")

            continue
        
        # try:
        #     image_path = os.path.join(args.input, args.mode, class_id, pic_name + '.JPEG')
        #     image_bgr = cv2.imread(image_path)
        #     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        #     mask_predictor.set_image(image_rgb)
        # except Exception:
        #     res = Image.fromarray(image.cpu().numpy().astype(np.uint8))
        #     res_path = os.path.join(args.output, class_id)
        #     if not os.path.exists(res_path):
        #         os.makedirs(res_path)
        #     res.save(os.path.join(res_path, pic_name + '.png'))
        #     now = datetime.now()
        #     processed_log.append(f"{len(processed_log)} {now.strftime('%Y-%m-%d_%H-%M-%S')}, {class_id}-{pic_name} failed to set this image as SAM source,  saved as original state")

        #     continue
        # 从伪标签取点作为SAM Prompt
        # index of non-zero points in psuedo-label picture, ordered in n * (W, H, D)
        # change it to n * (W, H)
        # non_zero = jt.nonzero(as_pseudo)[:, :2]
        # selected = non_zero[get_indices_from_distribution(non_zero.shape[0], args.distribution, args.points_count), :]
        # label = jt.ones([selected.shape[0]])
        # result, _, _ = mask_predictor.predict(selected.numpy(), label, multimask_output=False)

        # 将伪标签转化为框作为SAM Prompt
        # not_zero = jt.where(as_pseudo != 0)
        # x_min = int(min(not_zero[0]))
        # x_max = int(max(not_zero[0]))# * 1.3 if max(not_zero[0]) * 1.3 < image.shape[0] else image.shape[0])
        # y_min = int(min(not_zero[1]))
        # y_max = int(max(not_zero[1]))# * 1.3 if max(not_zero[1]) * 1.3 < image.shape[1] else image.shape[1])
        # #result, _, _ = mask_predictor.predict(box=np.array([x_min, y_min, x_max, y_max]), multimask_output=False)

        # # 从SaliencyRC输出取点作为SAM Prompt
        # saliency_result = get_rc(image_path)
        # saliency_result = remove_small_areas(saliency_result, 80)
        # zero_in_saliency = jt.where(saliency_result != 0)
        # possible_foreground = np.column_stack((zero_in_saliency[0], zero_in_saliency[1]))
        # selected_points = possible_foreground[get_indices_from_distribution(possible_foreground.shape[0], args.distribution, args.points_count), :]
        # foreground_labels = jt.ones([selected_points.shape[0]])

        # result, _, _ = mask_predictor.predict(selected_points, foreground_labels, box=np.array([x_min, y_min, x_max, y_max]), multimask_output=False)
        
        # output_path = os.path.join(args.output, class_id) 
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)
        
        # current_result = result[0, :, :]
        # current_result = jt.where(current_result != 0, max_label, 0)
        # result_genuine = jt.zeros([image.shape[0], image.shape[1], 3], dtype=jt.int32)
        # result_genuine[:, :, 0] = current_result[:, :]

        # res = Image.fromarray(result_genuine.cpu().numpy().astype(np.uint8))
        # res.save(os.path.join(output_path, pic_name + '.png'))
        # cv2.imwrite(os.path.join(output_path, pic_name + '-rc.png'), saliency_result)
        # now = datetime.now()
        # processed_log.append(f'{len(processed_log)} {now.strftime("%Y-%m-%d_%H-%M-%S")},{class_id}-{pic_name} finished')

        # mask_predictor.reset_image()

        current_sam_result_path = os.path.join(args.sam_result, args.mode, class_id, pic_name)
        if len(labels) != 0 and os.path.exists(current_sam_result_path):
            failed = False
            current_sam_results = [file for file in os.listdir(current_sam_result_path) if file.endswith('.png')]
            psuedo_to_ones = jt.where(as_pseudo != 0, 1, 0)
            area_A = jt.sum(psuedo_to_ones)

            new_result = jt.zeros([image.shape[0], image.shape[1], 1], dtype=jt.int32)
            applied_count = 0
            
            for path in current_sam_results:
                mask = cv2.imread(os.path.join(current_sam_result_path, path), cv2.IMREAD_GRAYSCALE)
                mask = np.where(mask > 128, 1, 0)
                try:
                    mask = mask.reshape(H, W, 1)
                except:
                    res = Image.fromarray(image.cpu().numpy().astype(np.uint8))
                    res_path = os.path.join(args.output, class_id)
                    if not os.path.exists(res_path):
                        os.makedirs(res_path)
                    res.save(os.path.join(res_path, pic_name + '.png'))
                    now = datetime.now()
                    processed_log.append(f'{len(processed_log)} {now.strftime("%Y-%m-%d_%H-%M-%S")},{class_id}-{pic_name} failed due to {path}')
                    
                    failed = True
                    break

                intersection = jt.logical_and(psuedo_to_ones, mask)
                area_B = jt.sum(mask)
                intersection_area = jt.sum(intersection)
                # ↓
                if intersection_area == 0 or area_B * 1.0 > area_A * 1.8:
                    # 无交集或当前mask比伪标签区域大太多，进入下一轮
                    continue
                
                # 计算相交面积占mask的比例
                intersection_ratio = intersection_area / min(area_A, area_B)
                if intersection_ratio >= 0.6:
                    mask = jt.where(mask == 1, True, False)
                    new_result[mask] = as_pseudo[mask]

                    applied_count += 1
            
            if failed:
                continue
            
            result = jt.zeros([image.shape[0], image.shape[1], 3], dtype=jt.int32)
            if applied_count != 0:
                result[:, :, 0] = new_result[:, :, 0]
            else:
                result[:, :, 0] = as_pseudo[:, : ,0]
            res = Image.fromarray(result.cpu().numpy().astype(np.uint8))
            res_path = os.path.join(args.output, class_id)

            if not os.path.exists(res_path):
                os.makedirs(res_path)
            res.save(os.path.join(res_path, pic_name + '.png'))
            now = datetime.now()
            processed_log.append(f'{len(processed_log)} {now.strftime("%Y-%m-%d_%H-%M-%S")},{class_id}-{pic_name} finished')
        else:
            # 判断类为空或者SAM分割执行错误，直接沿用旧结果
            res = Image.fromarray(image.cpu().numpy().astype(np.uint8))
            res_path = os.path.join(args.output, class_id)
            if not os.path.exists(res_path):
                os.makedirs(res_path)
            res.save(os.path.join(res_path, pic_name + '.png'))
            now = datetime.now()
            processed_log.append(f'{len(processed_log)} {now.strftime("%Y-%m-%d_%H-%M-%S")},{class_id}-{pic_name} finished')

    try:
        with open(os.path.join(args.output, 'log.txt'), 'w') as file:
            for item in processed_log:
                file.write(item + '\n')
    except IOError as e:
        print('failed writing log', e)

    print(f'integration finished.')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)