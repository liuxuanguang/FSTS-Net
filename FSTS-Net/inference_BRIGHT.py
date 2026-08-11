import sys
sys.path.append('/media/lenovo/课题研究/博士小论文数据/时空谱联合SCD/TSS-SCD')  # change this to the path of your project
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.MultiSiamese_RS_ST_TL_BRIGHT import MultimodalDamageAssessmentDatset
# from models.Proposed_method import mmscd_siam_GCN_WT_singleSTM as Net
# from models.HRSICD import HRSICD as Net
from models.Multimodal_SCD_0609 import mmscd_siam_GCN_WT_singleSTM as Net
from datetime import datetime
from utils.metrics import Evaluator
from PIL import Image
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Inference:
    def __init__(self, args):
        self.model_path = args.model_path
        self.output_dir = args.output_dir
        # config = get_config(args)
        num_classes = 4
        # Load dataset
        dataset = MultimodalDamageAssessmentDatset(args.test_dataset_path, args.test_data_list, 1024, None, 'train',
                                                   suffix='.tif')
        self.test_loader = DataLoader(dataset, batch_size=1, num_workers=8, drop_last=False)

        # Load model
        self.model = Net("resnet34", False, 4, "lightweight", 6, 0.00005)
        # self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(self.model_path), strict=False)
        self.model = self.model.cuda()
        self.model.eval()
        self.color_map = {
            0: (255, 255, 255),  # No damage - black
            1: (70, 181, 121),  # Minor damage - green
            2: (228, 189, 139),  # Major damage - yellow
            3: (182, 70, 69)  # Destroyed - red
        }
        # Overall evaluator
        self.evaluator = Evaluator(num_class=num_classes)
        self.single_evaluator = Evaluator(num_class=num_classes)
        self.evaluator_clf = Evaluator(num_class=num_classes)

        # Disaster-type-specific evaluators
        self.disaster_type_evaluator_dict = {event: Evaluator(num_class=num_classes) for event in
                                             self.get_disaster_types()}
        self.disaster_event_evaluator_dict = {event: Evaluator(num_class=num_classes) for event in
                                              self.get_disaster_events()}

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(os.path.join(self.output_dir, 'original')):
            os.makedirs(os.path.join(self.output_dir, 'original'))

        if not os.path.exists(os.path.join(self.output_dir, 'colored')):
            os.makedirs(os.path.join(self.output_dir, 'colored'))

    def get_disaster_events(self):
        """Returns a list of disaster events based on filename prefixes."""
        return [
            "turkey-earthquake", "hawaii-wildfire", "morocco-earthquake",
            "haiti-earthquake", "la_palma-volcano", "congo-volcano",
            "beirut-explosion", "bata-explosion", "libya-flood",
            "noto-earthquake", "marshall-wildfire", "ukraine-conflict", "myanmar-hurricane", "mexico-hurricane"
        ]

    def get_disaster_types(self):
        """Returns a list of disaster events based on filename prefixes."""
        return [
            "earthquake", "wildfire", "volcano", "explosion", "flood",
            "conflict", "hurricane"
        ]

    def apply_tta_inference(self, model, pre_change_imgs, post_change_imgs):
        """
        Performs test-time augmentations (TTA) on the input images and
        fuses the resulting logits. Returns fused logits for damage classification.

        Args:
            model (nn.Module): your model in eval mode
            pre_change_imgs (Tensor): shape [B, C, H, W]
            post_change_imgs (Tensor): shape [B, C, H, W]

        Returns:
            Tensor: fused logits with shape [B, num_damage_classes, H, W]
        """
        # Collect logits from each transform
        logits_collection = []

        # 1) No transform
        output_clf, output_bcd = model(pre_change_imgs, post_change_imgs)  # output_clf is [B, num_damage_classes, H, W]
        logits_collection.append(output_clf)

        # 2) Horizontal flip
        output_clf_hf, output_bcd_hf = model(pre_change_imgs.flip(dims=[3]), post_change_imgs.flip(dims=[3]))
        # Unflip the output back
        output_clf_hf = output_clf_hf.flip(dims=[3])
        logits_collection.append(output_clf_hf)

        # 3) Vertical flip
        output_clf_vf, output_bcd_vf = model(pre_change_imgs.flip(dims=[2]), post_change_imgs.flip(dims=[2]))
        # Unflip the output
        output_clf_vf = output_clf_vf.flip(dims=[2])
        logits_collection.append(output_clf_vf)

        # 4) 90-degree rotation
        # Note: torch.rot90(img, k, dims=(2,3)) rotates by 90*k degrees
        # dims=(2,3) => H, W
        pre_90 = torch.rot90(pre_change_imgs, 1, dims=(2, 3))
        post_90 = torch.rot90(post_change_imgs, 1, dims=(2, 3))
        output_clf_90, output_bcd_90 = model(pre_90, post_90)
        # invert rotation
        output_clf_90 = torch.rot90(output_clf_90, 3, dims=(2, 3))
        logits_collection.append(output_clf_90)

        # 5) 180-degree rotation
        pre_180 = torch.rot90(pre_change_imgs, 2, dims=(2, 3))
        post_180 = torch.rot90(post_change_imgs, 2, dims=(2, 3))
        output_clf_180, output_bcd_180 = model(pre_180, post_180)
        # invert rotation
        output_clf_180 = torch.rot90(output_clf_180, 2, dims=(2, 3))
        logits_collection.append(output_clf_180)

        # 6) 270-degree rotation
        pre_270 = torch.rot90(pre_change_imgs, 3, dims=(2, 3))
        post_270 = torch.rot90(post_change_imgs, 3, dims=(2, 3))
        output_clf_270, output_bcd_270 = model(pre_270, post_270)
        # invert rotation
        output_clf_270 = torch.rot90(output_clf_270, 1, dims=(2, 3))
        logits_collection.append(output_clf_270)

        # Fuse logits by averaging
        # shape: [B, num_damage_classes, H, W]
        fused_logits = torch.mean(torch.stack(logits_collection, dim=0), dim=0)

        return fused_logits

    def run_inference(self):
        print('Starting inference...')
        self.evaluator.reset()
        self.evaluator_loc = Evaluator(num_class=2)  # 用于位置变化检测的评估器
        self.evaluator_clf = Evaluator(num_class=4)  # 用于损伤分类的评估器

        with torch.no_grad():
            for i, data in enumerate(tqdm(self.test_loader)):
                self.single_evaluator.reset()
                pre_change_imgs, post_change_imgs, labels_loc, labels_clf, file_name = data
                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                file_name = file_name[0]  # Get the filename as a string

                # Predict
                output_clf, output_bcd = self.model(pre_change_imgs, post_change_imgs)

                # 处理输出
                output_clf = torch.argmax(output_clf, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                output_bcd = (output_bcd.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                labels_loc = labels_loc.squeeze().cpu().numpy()
                labels_clf = labels_clf.squeeze().cpu().numpy()

                # 只评估变化区域(标签>0的区域)
                output_clf_damage_part = output_clf[labels_loc > 0]
                labels_clf_damage_part = labels_clf[labels_loc > 0]

                # 添加batch到各个评估器
                self.evaluator_loc.add_batch(labels_loc, output_bcd)  # 位置变化检测评估
                self.evaluator_clf.add_batch(labels_clf_damage_part, output_clf_damage_part)  # 损伤分类评估
                self.evaluator.add_batch(labels_clf, output_clf)  # 整体评估

                # 保存结果
                self.save_colored_map(output_clf, file_name)
                self.save_original_map(output_clf, file_name)

                print(f'{file_name}: {self.single_evaluator.Mean_Intersection_over_Union()}')

                # 按灾害类型和事件分类评估
                for disaster_type in self.disaster_type_evaluator_dict.keys():
                    if disaster_type in file_name:
                        self.disaster_type_evaluator_dict[disaster_type].add_batch(labels_clf, output_clf)
                        break

                for event in self.disaster_event_evaluator_dict.keys():
                    if event in file_name:
                        self.disaster_event_evaluator_dict[event].add_batch(labels_clf, output_clf)
                        break

        # 计算各项指标
        loc_f1_score = self.evaluator_loc.Pixel_F1_score()  # 位置变化检测F1分数
        damage_f1_scores = self.evaluator_clf.Damage_F1_score()  # 各类别F1分数
        harmonic_mean_f1 = len(damage_f1_scores) / np.sum(1.0 / damage_f1_scores)  # 调和平均F1
        final_OA = self.evaluator.Pixel_Accuracy()  # 整体准确率
        mIoU = self.evaluator.Mean_Intersection_over_Union()  # 平均IoU
        IoU_of_each_class = self.evaluator.Intersection_over_Union()  # 各类别IoU

        # 打印结果
        print("\nFinal Evaluation Metrics:")
        print(f'Location Change F1 Score: {loc_f1_score:.4f}')
        print(f'Harmonic Mean F1 Score: {harmonic_mean_f1:.4f}')
        print(f'Overall Accuracy (OA): {final_OA:.4f}')
        print(f'Mean IoU: {mIoU:.4f}')
        print(f'Class-wise IoU: {[f"{i:.4f}" for i in IoU_of_each_class]}')
        print(f'Class-wise F1 Scores: {[f"{i:.4f}" for i in damage_f1_scores]}')

        # 计算并打印其他指标
        self.compute_and_print_overall_metrics()
        self.compute_and_print_disaster_event_metrics()
        self.compute_and_print_disaster_type_metrics()

        return {
            'loc_f1_score': loc_f1_score,
            'harmonic_mean_f1': harmonic_mean_f1,
            'final_OA': final_OA,
            'mIoU': mIoU,
            'IoU_of_each_class': IoU_of_each_class,
            'damage_f1_scores': damage_f1_scores
        }

    def save_original_map(self, prediction, file_name):
        """Saves the colored damage map."""
        # color_map_img = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        # for cls, color in self.color_map.items():
        #     color_map_img[prediction == cls] = color
        output_path = os.path.join(self.output_dir, 'original', file_name + '_building_damage.png')
        Image.fromarray(prediction).save(output_path)

    def save_colored_map(self, prediction, file_name):
        """Saves the colored damage map."""
        color_map_img = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        for cls, color in self.color_map.items():
            color_map_img[prediction == cls] = color
        output_path = os.path.join(self.output_dir, 'colored', file_name + '_building_damage.png')
        Image.fromarray(color_map_img).save(output_path)

    def compute_and_print_overall_metrics(self):
        """Computes and prints overall metrics."""

        pixel_accuracy = self.evaluator.Pixel_Accuracy()
        mean_iou = self.evaluator.Mean_Intersection_over_Union()
        print("\nOverall Metrics:")
        print(f'Pixel Accuracy: {pixel_accuracy * 100:.2f}%')
        print(f'Mean IoU: {mean_iou * 100:.2f}%')
        print(f'IoU: {self.evaluator.Intersection_over_Union()}')
        # print(f'F1 Score: {len(self.evaluator_clf.Damage_F1_socore()) / np.sum(1.0 / self.evaluator_clf.Damage_F1_socore()) * 100}')

    def compute_and_print_disaster_type_metrics(self):
        """Computes and prints mIoU for each disaster type."""
        print("\nPer-Disaster Type mIoU:")
        average_mIoU = 0
        for disaster_type, evaluator in self.disaster_type_evaluator_dict.items():
            mean_iou = evaluator.Mean_Intersection_over_Union()
            iou_per_class = evaluator.Intersection_over_Union()
            average_mIoU += mean_iou
            print(f"{disaster_type}: mIoU = {mean_iou * 100:.2f}%, IoU = {iou_per_class * 100}")
        print(f"Average mIoU = {average_mIoU / 7 * 100:.2f}%")

    def compute_and_print_disaster_event_metrics(self):
        """Computes and prints mIoU for each disaster event."""
        print("\nPer-Event Type mIoU:")
        average_mIoU = 0
        for event, evaluator in self.disaster_event_evaluator_dict.items():
            mean_iou = evaluator.Mean_Intersection_over_Union()
            iou_per_class = evaluator.Intersection_over_Union()
            average_mIoU += mean_iou
            print(f"{event}: mIoU = {mean_iou * 100:.2f}%, IoU = {iou_per_class * 100}")
        print(f"Average mIoU = {average_mIoU / 14 * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on BRIGHT")
    parser.add_argument('--model_path', type=str, default='/*****/model_weight_Bright.pth')
    parser.add_argument('--test_dataset_path', type=str, default='/*****/BRIGHT')
    parser.add_argument('--test_data_list_path', type=str, default='/*****/BRIGHT/test_set.txt')
    parser.add_argument('--output_dir', type=str, default='/*****/*****')

    args = parser.parse_args()

    # Load test data list
    with open(args.test_data_list_path, "r") as f:
        test_data_list = [data_name.strip() for data_name in f]
    args.test_data_list = test_data_list

    inference = Inference(args)
    inference.run_inference()
