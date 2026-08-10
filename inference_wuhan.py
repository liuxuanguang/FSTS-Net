import os
import PIL.Image as Image

# from models.Multimodal_SCD import mmscd_siam_DynamicGCN_wuhan as Net
from datasets.MultiSiamese_RS_ST_TL_BRIGHT_BCD import MultimodalDamageAssessmentDatset
from models.Proposed_method import mmscd_siam_GCN_WT_singleSTM_WH as Net
# from models.DamageFormer import DamageFormer_wuhan as Net
# from models.SiamAttnUNet import SiamAttnUNet as Net
# from models.HGINet import HGINet_wuhan as Net
# from models.Multimodal_SCD import mmscd_sigma_wuhan as Net
# from models.Multimodal_SCD import mmscd_siam_DynamicGCN_wuhan as Net
import os
import numpy as np
import torch
from utils.palette import color_map
from utils.metric import IOUandSek
from tqdm import tqdm
from torch.utils.data import DataLoader
from thop import profile
import time
import argparse


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('Building Change Detection')
        parser.add_argument("--data_name", type=str, default="Wuhan")
        parser.add_argument("--Net_name", type=str, default="proposed_wuhan")
        parser.add_argument("--lightweight", dest="lightweight", action="store_true",
                            help='lightweight head for fewer parameters and faster speed')
        parser.add_argument("--backbone", type=str, default="resnet34")
        parser.add_argument("--data_root", type=str, default=r"/******/wuhan_data")
        parser.add_argument("--test_data_name_list", type=str, default=r"/******/wuhan_data/test_set.txt")
        parser.add_argument("--load_from", type=str,
                            default=r"/*****/model_weight_wuhan.pth")
        parser.add_argument("--test_batch_size", type=int, default=16)
        parser.add_argument("--pretrained", type=bool, default=True,
                            help='initialize the backbone with pretrained parameters')
        parser.add_argument("--tta", dest="tta", action="store_true",
                            help='test_time_augmentation')
        parser.add_argument("--M", type=int, default=6)
        parser.add_argument("--Lambda", type=float, default=0.00005)
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        print(args)
        return args


def safe_load_weights(model, checkpoint_path, device):
    """安全加载权重函数"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"加载权重文件: {checkpoint_path}")
        print(f"权重文件键数量: {len(checkpoint.keys())}")

        # 检查模型和权重文件的键匹配情况
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint.keys())

        print(f"模型参数数量: {len(model_keys)}")
        print(f"匹配的参数: {len(model_keys.intersection(checkpoint_keys))}")
        print(f"缺失的参数: {model_keys - checkpoint_keys}")
        print(f"多余的参数: {checkpoint_keys - model_keys}")

        # 尝试严格加载
        try:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=True)
            print("✓ 严格加载成功")
            if missing_keys:
                print(f"缺失的键: {missing_keys}")
            if unexpected_keys:
                print(f"意外的键: {unexpected_keys}")
            return True
        except Exception as e:
            print(f"严格加载失败: {e}")
            print("尝试非严格加载...")

            # 非严格加载
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            print("✓ 非严格加载成功")
            if missing_keys:
                print(f"缺失的键: {len(missing_keys)}个")
            if unexpected_keys:
                print(f"意外的键: {len(unexpected_keys)}个")
            return True

    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return False

def inference(args):
    begin_time = time.time()
    working_path = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(working_path, 'pred_results', args.data_name, args.Net_name, args.backbone)
    pred_save_path3 = os.path.join(pred_dir, 'pred_change')

    if not os.path.exists(pred_save_path3): os.makedirs(pred_save_path3)

    testset = MultimodalDamageAssessmentDatset(args.data_root, args.test_data_name_list, crop_size=256, max_iters=None,
                                               type='test')
    testloader = DataLoader(testset, batch_size=8, shuffle=True, num_workers=8, drop_last=False)

    # testset = ChangeDetection_LEVIR_CD(root=args.data_root, mode="test")
    # testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
    #                             pin_memory=True, num_workers=0, drop_last=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(backbone="resnet34", pretrained=False, nclass=1, lightweight="lightweight", M=6, Lambda=0.00005)
    # model = Net(in_channels=3, num_classes=1)
    model = model.to(device)
    # model = Net(args.backbone, args.pretrained, len(ChangeDetection_LEVIR_CD.CLASSES)-1, args.lightweight, args.M, args.Lambda)

    if args.load_from:
        model.load_state_dict(torch.load(args.load_from, map_location=device), strict=True)

    # # 安全加载权重
    # if args.load_from and os.path.exists(args.load_from):
    #     safe_load_weights(model, args.load_from, device)
    # else:
    #     print(f"❌ 权重文件不存在: {args.load_from}")
    #     return

    model.eval()

    # calculate Pamrams and FLOPs
    for vi, data in enumerate(testloader):
        if vi == 0:
            img1, img2, _, id = data
            img1, img2 = img1.to(device).float(), img2.to(device).float()
            break
    # input = torch.cat([img1, img2], dim=1).to(device).float()
    # FLOPs, Params = profile(model, (input,))
    FLOPs, Params = profile(model, (img1, img2))
    print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))

    tbar = tqdm(testloader)
    metric = IOUandSek(num_classes=2)
    with torch.no_grad():
        for img1, img2, label1, id in tbar:
            img1, img2 = img1.to(device).float(), img2.to(device).float()
            # input_data = torch.cat([img1, img2], dim=1)
            # out_bn = model(input_data)
            out_bn = model(img1, img2)


        #yujie_xiugai
            # ========== 修复：形状处理 ==========
            # 确保应用sigmoid（如果模型没有）
            if out_bn.min() < 0 or out_bn.max() > 1:
                out_bn = torch.sigmoid(out_bn)

            # 调整形状：确保是 (batch_size, height, width)
            if out_bn.dim() == 4:
                if out_bn.size(1) == 1:
                    # (B, 1, H, W) -> (B, H, W)
                    out_bn = out_bn.squeeze(1)
                else:
                    # 如果有多个通道，取第一个通道
                    out_bn = out_bn[:, 0, :, :]

            # 二值化并转换为numpy
            out_bn_np = (out_bn.cpu().numpy() > 0.5).astype(np.uint8)

            print(f"预测结果形状: {out_bn_np.shape}")  # 调试信息

            # 保存图像
            for i in range(out_bn_np.shape[0]):
                mask_data = out_bn_np[i]

                # 最终形状检查
                if mask_data.ndim != 2:
                    # 如果是3维，取最后两个维度
                    mask_data = mask_data.reshape(mask_data.shape[-2], mask_data.shape[-1])

                # 创建图像
                mask_img = Image.fromarray(mask_data * 255, mode='L')

                filename = id[i]

            # out_bn = ((out_bn > 0.5).cpu().numpy()).astype(np.uint8)
            # # out_bn = out_bn.squeeze(1)
            # cmap = color_map()
            #
            # for i in range(out_bn.shape[0]):
            #
            #     mask_bn = Image.fromarray(out_bn[i] * 255)
            #     filename = id[i]
                if not filename.lower().endswith('.png'):
                    filename += '.png'

                mask_img.save(os.path.join(pred_save_path3, filename))

            metric.add_batch(out_bn_np, label1.numpy())

        # Recall, Precision, OA, F1, IoU, KC = metric.evaluate_BCD()
        Recall, Precision, OA, F1, IoU, mIoU, KC = metric.evaluate_BCD()

        print('==>Recall', Recall)
        print('==>Precision', Precision)
        print('==>OA', OA)
        print('==>F1', F1)
        print('==>IoU', IoU)
        print('==>KC', KC)
        print('==>mIoU', mIoU)

        time_use = time.time() - begin_time

    metric_file = os.path.join(pred_dir, 'metric.txt')
    f = open(metric_file, 'w', encoding='utf-8')
    f.write("Data：" + str(args.data_name) + '\n')
    f.write("model：" + str(args.Net_name) + '\n')
    f.write("##################### metric #####################" + '\n')
    f.write("infer time (s) ：" + str(round(time_use, 2)) + '\n')
    f.write("Params (Mb) ：" + str(round(Params / 1e6, 2)) + '\n')
    f.write("FLOPs (Gbps) ：" + str(round(FLOPs / 1e9, 2)) + '\n')
    f.write('\n')
    f.write("Recall (%) ：" + str(round(Recall * 100, 2)) + '\n')
    f.write("Precision (%) ：" + str(round(Precision * 100, 2)) + '\n')
    f.write("OA (%) ：" + str(round(OA * 100, 2)) + '\n')
    f.write("F1 (%) ：" + str(round(F1 * 100, 2)) + '\n')
    f.write("IoU (%) ：" + str(round(IoU * 100, 2)) + '\n')
    f.write("KC (%) ：" + str(round(KC * 100, 2)) + '\n')
    f.write("mIoU (%) ：" + str(round(mIoU * 100, 2)) + '\n')

    f.close()


if __name__ == "__main__":
    args = Options().parse()
    inference(args)