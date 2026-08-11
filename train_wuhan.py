import sys
sys.path.append('/data/FSTS-Net')  # change this to the path of your project
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.MultiSiamese_RS_ST_TL_BRIGHT_BCD import MultimodalDamageAssessmentDatset
from models.Proposed_method import mmscd_siam_GCN_WT_singleSTM_WH as Net
# from models.Multimodal_SCD1 import mmscd_CL_GCN_V2 as Net
from utils.metrics import Evaluator
from datetime import datetime
from torch.nn import CrossEntropyLoss, BCELoss
from utils.loss import DiceLoss
import utils.lovasz_loss as L
import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class SimilarityLoss(nn.Module):
    """
    相似损失函数：强制两个时相的共享特征保持一致

    数学公式：
        L_sim = 1 - cos(z_c^{t1}, z_c^{t2})
               = 1 - \frac{z_c^{t1} \cdot z_c^{t2}}{\|z_c^{t1}\| \cdot \|z_c^{t2}\|}

    特点：
      - 余弦相似度范围[-1,1]，映射为损失值[0,2]
      - 对特征幅度变化不敏感，专注方向一致性
      - 特别适合消除季节/光照变化影响
    """

    def __init__(self, reduction='mean'):
        super(SimilarityLoss, self).__init__()
        self.reduction = reduction

    def forward(self, feat_t1, feat_t2):
        # 输入特征尺寸: (B, C, H, W) 或 (B, C)
        batch_size = feat_t1.size(0)

        # 展平特征为向量 (B, C*H*W)
        feat_t1_flat = feat_t1.reshape(batch_size, -1)
        feat_t2_flat = feat_t2.reshape(batch_size, -1)

        # 计算余弦相似度矩阵 (B, B) -> 取对角线
        cosine_matrix = F.cosine_similarity(
            feat_t1_flat.unsqueeze(1),
            feat_t2_flat.unsqueeze(0),
            dim=2
        )
        cos_sim = torch.diag(cosine_matrix)  # (B,)

        # 余弦相似度 -> 损失值 [0,2]
        loss = 1.0 - cos_sim

        # 批次平均
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DifferenceLoss(nn.Module):
    """
    差异损失函数：强制共享特征与私有特征在向量空间正交

    数学公式：
        L_diff = \| Z_c^T Z_p \|_F^2
                = \sum_{i,j} |(z_c^{(i)})^T z_p^{(j)}|^2

    特点：
      - 通过Frobenius范数约束特征正交性
      - 防止信息冗余，明确分工共享/私有特征
      - 支持多种实现模式(全局/局部)
    """
    def __init__(self, mode='global', reduction='mean'):
        """
        Args:
            mode: 'global' - 使用全局向量计算
                   'local' - 按空间位置计算(保持原分辨率)
            reduction: 'mean' | 'sum' | 'none'
        """
        super(DifferenceLoss, self).__init__()
        self.mode = mode
        self.reduction = reduction

    def forward(self, shared_feat, private_feat):
        """
        输入:
          shared_feat: 共享特征 (B, C_s, H, W)
          private_feat: 私有特征 (B, C_p, H, W)
        """
        if self.mode == 'global':
            return self._global_orthogonality(shared_feat, private_feat)
        else:  # local
            return self._local_orthogonality(shared_feat, private_feat)

    def _global_orthogonality(self, shared_feat, private_feat):
        # 全局平均池化 -> 向量 (B, C_s) 和 (B, C_p)
        shared_vec = F.adaptive_avg_pool2d(shared_feat, 1).squeeze(-1).squeeze(-1)  # (B, C_s)
        private_vec = F.adaptive_avg_pool2d(private_feat, 1).squeeze(-1).squeeze(-1)  # (B, C_p)

        # 计算点积矩阵 (B, C_s, C_p)
        dot_product = torch.einsum('bc,bd->bcd', shared_vec, private_vec)

        # Frobenius范数平方 (B,)
        loss_per_sample = torch.norm(dot_product, p='fro', dim=(1, 2)) ** 2

        return self._reduce_loss(loss_per_sample)

    def _local_orthogonality(self, shared_feat, private_feat):
        # 提取特征维度
        B, C_s, H, W = shared_feat.shape
        _, C_p, _, _ = private_feat.shape

        # 重排维度: (B, C, H, W) -> (B, H, W, C)
        shared_perm = shared_feat.permute(0, 2, 3, 1)  # (B, H, W, C_s)
        private_perm = private_feat.permute(0, 2, 3, 1)  # (B, H, W, C_p)

        # 计算位置相关度矩阵 (B, H, W, C_s, C_p)
        dot_product = torch.einsum('bhwc,bhwd->bhwcd', shared_perm, private_perm)

        # Frobenius范数平方 (B, H, W)
        loss_per_pixel = torch.norm(dot_product, p='fro', dim=(-2, -1)) ** 2

        # 空间位置平均 (B,)
        loss_per_sample = loss_per_pixel.mean(dim=(1, 2))

        return self._reduce_loss(loss_per_sample)

    def _reduce_loss(self, loss_per_sample):
        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else:  # 'none'
            return loss_per_sample


class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()

    def forward(self, p, q):
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        loss = F.kl_div(q.log(), p, reduction='batchmean')
        return loss

class Trainer(object):
    """
    Trainer class that encapsulates model, optimizer, and data loading.
    It can train the model and evaluate its performance on a holdout set.
    """
    def __init__(self, args):
        """
        Initialize the Trainer with arguments from the command line or defaults.

        :param args: Argparse namespace containing:
            - dataset, train_dataset_path, holdout_dataset_path, etc.
            - model_type, model_param_path, resume path for checkpoint
            - learning rate, weight decay, etc.
        """
        self.args = args
        # Initialize evaluator for metrics such as accuracy, IoU, etc.
        self.evaluator_loc = Evaluator(num_class=2)

        # Create the deep learning model. Here we show how to use UNet or SiamCRNN.
        self.deep_model = Net(backbone="resnet34", pretrained=False, nclass=1, lightweight="lightweight", M=6, Lambda=0.00005)
        # self.deep_model = Net(in_channels=3, num_classes=1)
        self.deep_model = self.deep_model.cuda()

        # Create a directory to save model weights, organized by timestamp.
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_path = os.path.join(args.model_param_path, args.dataset, args.model_type + '_' + now_str)

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)
        self.criterion_bn = BCELoss(reduction='none')
        self.criterion_bn_2 = DiceLoss()
        # self.kl = KLDivLoss()
        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    def training(self):
        """
        Main training loop that iterates over the training dataset for several steps (max_iters).
        Prints intermediate losses and evaluates on holdout dataset periodically.
        """
        best_mIoU = 0.0
        total_loss_bn = 0.0
        # kl_loss = 0.0
        best_round = []
        torch.cuda.empty_cache()
        train_dataset = MultimodalDamageAssessmentDatset(self.args.train_dataset_path, self.args.train_data_name_list,
                                                         crop_size=self.args.crop_size, max_iters=self.args.max_iters,
                                                         type='train')
        train_data_loader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, shuffle=True,
                                       num_workers=self.args.num_workers, drop_last=False)
        elem_num = len(train_data_loader)
        train_enumerator = enumerate(train_data_loader)
        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            pre_change_imgs, post_change_imgs, labels_loc, _ = data
            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            labels_loc = labels_loc.cuda().float()
            out_bcd = self.deep_model(pre_change_imgs, post_change_imgs)  # if you use UNet
            # feature1, feature2, out_bcd = self.deep_model(pre_change_imgs, post_change_imgs)  # if you use UNet
            # outout_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs) # If you use SiamCRNN
            self.optim.zero_grad()
            # ce_loss_loc = F.cross_entropy(outout_loc, labels_loc, ignore_index=255) # if you use SiamCRNN
            # lovasz_loss_loc = L.lovasz_softmax(F.softmax(outout_loc, dim=1), labels_loc, ignore=255) # if you use SiamCRNN

            loss_bn_1 = self.criterion_bn(out_bcd, labels_loc)
            loss_bn_1[labels_loc == 1] *= 2
            loss_bn_1 = loss_bn_1.mean()
            loss_bn_2 = self.criterion_bn_2(out_bcd, labels_loc)

            # loss_mf = self.kl(feature1, feature2)
            loss_bn = loss_bn_1 + loss_bn_2
            total_loss_bn += loss_bn.item()
            loss_bn.backward()

            self.optim.step()

            if (itera + 1) % 100 == 0:
                print(f'iter is {itera + 1}, change loss is {loss_bn.item()}')
                if (itera + 1) % 500 == 0:
                    self.deep_model.eval()
                    loc_f1_score_val, OA_val, Precision_val, mIoU_val, Recall_val, mIoU_val = self.validation()
                    loc_f1_score_test, OA_test, Precision_test, mIoU_test, Recall_test, mIoU_test = self.test()

                    if mIoU_test > best_mIoU:
                        save_name = f'best_model_val_{mIoU_val * 100:.1f}_test_{mIoU_test * 100:.1f}.pth'  # 保留1位小数格式
                        torch.save(self.deep_model.state_dict(), os.path.join(self.model_save_path, save_name))
                        best_mIoU = mIoU_test
                        best_round = {
                            'best iter': itera + 1,
                            'loc f1 (val)': loc_f1_score_val * 100,
                            'OA (val)': OA_val * 100,
                            'mIoU (val)': mIoU_val * 100,
                            'loc f1 (test)': loc_f1_score_test * 100,
                            'OA (test)': OA_val * 100,
                            'mIoU (test)': mIoU_test * 100
                        }
                    self.deep_model.train()

        print('The accuracy of the best round is ', best_round)

    def validation(self):
        print('---------starting validation-----------')
        self.evaluator_loc.reset()
        val_dataset = MultimodalDamageAssessmentDatset(self.args.val_dataset_path, self.args.val_data_name_list, 256,
                                                       None, 'test')
        val_data_loader = DataLoader(val_dataset, batch_size=self.args.eval_batch_size, num_workers=1, drop_last=False)
        torch.cuda.empty_cache()

        with torch.no_grad():
            for _, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels_loc, _ = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_loc = labels_loc.cuda().long()

                output_loc = self.deep_model(pre_change_imgs, post_change_imgs)  # if you use UNet
                # feature1, feature2, output_loc = self.deep_model(pre_change_imgs, post_change_imgs)  # if you use UNet
                # _, output_clf = self.deep_model(pre_change_imgs, post_change_imgs) # If you use SiamCRNN

                labels_loc = labels_loc.cpu().numpy()

                output_loc = output_loc.detach().cpu().numpy()
                output_loc = (output_loc >= 0.5).astype(np.uint8)  # 直接二值化

                self.evaluator_loc.add_batch(labels_loc, output_loc)

        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        OA = self.evaluator_loc.Pixel_Accuracy()
        Precision = self.evaluator_loc.Pixel_Precision_Rate()
        Recall = self.evaluator_loc.Pixel_Recall_Rate()
        mIoU = self.evaluator_loc.Mean_Intersection_over_Union()
        print(f'loc_f1_score is {100 * loc_f1_score}, OA is {100 * OA}, mIoU is {100 * mIoU}, Precision is {100 * Precision}, Recall is {100 * Recall}')
        return loc_f1_score, OA, Precision, mIoU, Recall, mIoU

    def test(self):
        print('---------starting testing-----------')
        self.evaluator_loc.reset()
        test_dataset = MultimodalDamageAssessmentDatset(self.args.test_dataset_path, self.args.test_data_name_list,
                                                        256, None, 'test')
        test_data_loader = DataLoader(test_dataset, batch_size=self.args.eval_batch_size, num_workers=1,
                                      drop_last=False)
        torch.cuda.empty_cache()

        with torch.no_grad():
            for _, data in enumerate(test_data_loader):
                pre_change_imgs, post_change_imgs, labels_loc, _ = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_loc = labels_loc.cuda().long()

                output_loc = self.deep_model(pre_change_imgs, post_change_imgs)  # if you use UNet
                # feature1, feature2, output_loc = self.deep_model(pre_change_imgs, post_change_imgs)  # if you use UNet
                # _, output_clf = self.deep_model(pre_change_imgs, post_change_imgs) # If you use SiamCRNN

                labels_loc = labels_loc.cpu().numpy()
                output_loc = output_loc.detach().cpu().numpy()
                output_loc = (output_loc >= 0.5).astype(np.uint8)  # 直接二值化
                self.evaluator_loc.add_batch(labels_loc, output_loc)

        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        OA = self.evaluator_loc.Pixel_Accuracy()
        Precision = self.evaluator_loc.Pixel_Precision_Rate()
        Recall = self.evaluator_loc.Pixel_Recall_Rate()
        mIoU = self.evaluator_loc.Mean_Intersection_over_Union()
        print(f'loc_f1_score is {100 * loc_f1_score}, OA is {100 * OA}, mIoU is {100 * mIoU}, Precision is {100 * Precision}, Recall is {100 * Recall}')
        return loc_f1_score, OA, Precision, mIoU, Recall, mIoU


def main():
    parser = argparse.ArgumentParser(description="Training on BRIGHT dataset")
    parser.add_argument('--dataset', type=str, default='Wuhan')
    parser.add_argument('--train_dataset_path', type=str, default='/*****/wuhan_data')
    parser.add_argument('--train_data_list_path', type=str, default='/*****/wuhan_data/train_set.txt')
    parser.add_argument('--val_dataset_path', type=str, default='/*****/wuhan_data')
    parser.add_argument('--val_data_list_path', type=str, default='/*****/wuhan_data/val_set.txt')
    parser.add_argument('--test_dataset_path', type=str, default='/*****/wuhan_data')
    parser.add_argument('--test_data_list_path', type=str, default='/*****/wuhan_data/test_set.txt')
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--val_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=600000)
    parser.add_argument('--model_type', type=str, default='wuhan')
    parser.add_argument('--model_param_path', type=str,
                        default='/mnt/nas/checkpoints')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()

    with open(args.train_data_list_path, "r") as f:
        train_data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = train_data_name_list

    with open(args.val_data_list_path, "r") as f:
        val_data_name_list = [data_name.strip() for data_name in f]
    args.val_data_name_list = val_data_name_list

    with open(args.test_data_list_path, "r") as f:
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()