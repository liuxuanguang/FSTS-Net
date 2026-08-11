import torch
import torch.distributed as dist

from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from einops import rearrange

from tree_scan import _C

class _MST(Function):
    @staticmethod
    def forward(ctx, edge_index, edge_weight, vertex_index):
        edge_out = _C.mst_forward(edge_index, edge_weight, vertex_index)
        return edge_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return None, None, None

mst = _MST.apply

def norm2_distance(fm_ref, fm_tar):
    diff = fm_ref - fm_tar
    weight = (diff * diff).sum(dim=1)
    return weight

def norm1_distance(fm_ref, fm_tar):
    diff = fm_ref - fm_tar
    weight = torch.abs(diff).sum(dim=1)
    return weight

class MinimumSpanningTree(nn.Module):
    def __init__(self, distance_func, mapping_func=None):
        super(MinimumSpanningTree, self).__init__()
        self.distance_func = distance_func
        self.mapping_func = mapping_func
    
    @staticmethod
    def _build_matrix_index(fm):
        batch, height, width = (fm.shape[0], *fm.shape[2:])
        row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
        raw_index = row + col * width
        row_index = torch.stack([raw_index[:-1, :], raw_index[1:, :]], 2)
        col_index = torch.stack([raw_index[:, :-1], raw_index[:, 1:]], 2)
        index = torch.cat([row_index.reshape(1, -1, 2), 
                           col_index.reshape(1, -1, 2)], 1)
        index = index.expand(batch, -1, -1)
        return index

    def _build_feature_weight(self, fm):
        batch = fm.shape[0]
        weight_row = norm2_distance(fm[:, :, :-1, :], fm[:, :, 1:, :])
        weight_col = norm2_distance(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        weight = torch.cat([weight_row, weight_col], dim=1)
        if self.mapping_func is not None:
            weight = self.mapping_func(weight)
        return weight

    def _build_feature_weight_cosine(self, fm, max_tree):
        batch,dim = fm.shape[0],fm.shape[1]
        weight_row = torch.cosine_similarity(fm[:, :, :-1, :].reshape(batch,dim,-1), fm[:, :, 1:, :].reshape(batch,dim,-1),dim=1)
        # import pdb;pdb.set_trace()
        weight_col = torch.cosine_similarity(fm[:, :, :, :-1].reshape(batch,dim,-1), fm[:, :, :, 1:].reshape(batch,dim,-1),dim=1)
        weight = torch.cat([weight_row, weight_col], dim=1)
        if self.mapping_func is not None:
            if max_tree:
                weight = self.mapping_func(weight)   # cosine similarity needs "-weight" for min tree, "weight" for max tree
            else:
                weight = self.mapping_func(-weight)   # cosine similarity needs "-weight" for min tree, "weight" for max tree
        return weight

    def forward(self, guide_in, max_tree=False):
        with torch.no_grad():
            index = self._build_matrix_index(guide_in)
            if self.distance_func == "Cosine":
                weight = self._build_feature_weight_cosine(guide_in,max_tree)
            else:
                weight = self._build_feature_weight(guide_in)
            tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3])
            # tree = mst(index, weight, guide_in.shape[2])
        return tree


class MinimumSpanning3DTree(nn.Module):
    def __init__(self, distance_func, mapping_func=None):
        super(MinimumSpanning3DTree, self).__init__()
        self.distance_func = distance_func
        self.mapping_func = mapping_func

    @staticmethod
    def _build_matrix_index(fm):
        batch, height, width = (fm.shape[0], *fm.shape[2:])
        row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
        raw_index = row + col * width
        # 修改后代码
        # 获取左半部分和右半部分的索引
        mid = width // 2
        left_raw_index = raw_index[:, :mid]
        right_raw_index = raw_index[:, mid:]

        # 获取左半部分的row_index和col_index
        left_row_index = torch.stack([left_raw_index[:-1, :], left_raw_index[1:, :]], 2)
        left_col_index = torch.stack([left_raw_index[:, :-1], left_raw_index[:, 1:]], 2)

        # 获取右半部分的row_index和col_index
        right_row_index = torch.stack([right_raw_index[:-1, :], right_raw_index[1:, :]], 2)
        right_col_index = torch.stack([right_raw_index[:, :-1], right_raw_index[:, 1:]], 2)

        # 获取左半部分和右半部分对应位置的索引
        # cross_row_index = torch.stack([left_raw_index[:-1, :], right_raw_index[1:, :]], 2)
        # 训练在512尺寸数据集
        cross_col_index = torch.stack([left_raw_index[:, :], right_raw_index[:, :]], 2)
        # 训练在400尺寸的3DCD数据集
        # cross_height = min(left_raw_index.shape[0], right_raw_index.shape[0])
        # cross_col_index = torch.stack([
        #     left_raw_index[:cross_height, :],
        #     right_raw_index[:cross_height, :]
        # ], 2)

        # 合并所有索引
        index = torch.cat([
            left_row_index.reshape(1, -1, 2),
            left_col_index.reshape(1, -1, 2),
            right_row_index.reshape(1, -1, 2),
            right_col_index.reshape(1, -1, 2),
            cross_col_index.reshape(1, -1, 2)
        ], 1)

        index = index.expand(batch, -1, -1)

        return index

    def _build_feature_weight(self, fm):
        batch = fm.shape[0]
        weight_row = norm2_distance(fm[:, :, :-1, :], fm[:, :, 1:, :])
        weight_col = norm2_distance(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weihth_bi = norm2_distance(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        weight_bi = weihth_bi.reshape([batch, -1])
        weight = torch.cat([weight_row, weight_col, weight_bi], dim=1)
        if self.mapping_func is not None:
            weight = self.mapping_func(weight)
        return weight

    def _build_feature_weight_cosine(self, fm, max_tree):
        batch, dim = fm.shape[0], fm.shape[1]
        half_width = fm.shape[3] // 2
        fm_left_half = fm[:, :, :, :half_width]
        fm_right_half = fm[:, :, :, half_width:]
        # 左边weight_row和weight_col
        left_weight_row = torch.cosine_similarity(fm_left_half[:, :, :-1, :].reshape(batch, dim, -1),
                                             fm_left_half[:, :, 1:, :].reshape(batch, dim, -1), dim=1)
        left_weight_col = torch.cosine_similarity(fm_left_half[:, :, :, :-1].reshape(batch, dim, -1),
                                             fm_left_half[:, :, :, 1:].reshape(batch, dim, -1), dim=1)

        # 右边weight_row和weight_col
        right_weight_row = torch.cosine_similarity(fm_right_half[:, :, :-1, :].reshape(batch, dim, -1),
                                             fm_right_half[:, :, 1:, :].reshape(batch, dim, -1), dim=1)
        right_weight_col = torch.cosine_similarity(fm_right_half[:, :, :, :-1].reshape(batch, dim, -1),
                                             fm_right_half[:, :, :, 1:].reshape(batch, dim, -1), dim=1)

        # 左右对应位置权重
        weight_bi = torch.cosine_similarity(fm_left_half.reshape(batch, dim, -1),
                                             fm_right_half.reshape(batch, dim, -1), dim=1)


        weight = torch.cat([left_weight_row, left_weight_col, right_weight_row, right_weight_col, weight_bi], dim=1)
        if self.mapping_func is not None:
            if max_tree:
                weight = self.mapping_func(
                    weight)  # cosine similarity needs "-weight" for min tree, "weight" for max tree
            else:
                weight = self.mapping_func(
                    -weight)  # cosine similarity needs "-weight" for min tree, "weight" for max tree
        return weight

    def forward(self, guide_in, max_tree=False):
        with torch.no_grad():
            index = self._build_matrix_index(guide_in)
            if self.distance_func == "Cosine":
                weight = self._build_feature_weight_cosine(guide_in, max_tree)
            else:
                weight = self._build_feature_weight(guide_in)
            tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3])
            # tree = mst(index, weight, guide_in.shape[2])
        return tree


class MinimumSpanningMT3DTree(nn.Module):
    def __init__(self, distance_func, mapping_func=None, Tem=3):
        super(MinimumSpanningMT3DTree, self).__init__()
        self.distance_func = distance_func
        self.mapping_func = mapping_func

    @staticmethod
    def _build_matrix_index(fm, Tem=3):
        batch, height, width = (fm.shape[0], *fm.shape[2:])
        row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
        raw_index = row + col * width
        # 修改后代码
        # 获取左半部分和右半部分的索引
        mid = width // Tem
        T1_raw_index = raw_index[:, :mid]
        T2_raw_index = raw_index[:, mid:2*mid]
        T3_raw_index = raw_index[:, 2*mid:3*mid]
        # 获取T1的row_index和col_index
        T1_row_index = torch.stack([T1_raw_index[:-1, :], T1_raw_index[1:, :]], 2)
        T1_col_index = torch.stack([T1_raw_index[:, :-1], T1_raw_index[:, 1:]], 2)

        # 获取T2的row_index和col_index
        T2_row_index = torch.stack([T2_raw_index[:-1, :], T2_raw_index[1:, :]], 2)
        T2_col_index = torch.stack([T2_raw_index[:, :-1], T2_raw_index[:, 1:]], 2)

        # 获取T3的row_index和col_index
        T3_row_index = torch.stack([T3_raw_index[:-1, :], T3_raw_index[1:, :]], 2)
        T3_col_index = torch.stack([T3_raw_index[:, :-1], T3_raw_index[:, 1:]], 2)
        # 获取左半部分和右半部分对应位置的索引
        # cross_row_index = torch.stack([left_raw_index[:-1, :], right_raw_index[1:, :]], 2)
        cross_col_index = torch.stack([T1_raw_index[:, :], T2_raw_index[:, :]], 2)

        # 合并所有索引
        index = torch.cat([
            T1_row_index.reshape(1, -1, 2),
            T1_col_index.reshape(1, -1, 2),
            T2_row_index.reshape(1, -1, 2),
            T2_col_index.reshape(1, -1, 2),
            T3_row_index.reshape(1, -1, 2),
            T3_col_index.reshape(1, -1, 2),
            cross_col_index.reshape(1, -1, 2)
        ], 1)

        index = index.expand(batch, -1, -1)

        return index

    def _build_feature_weight(self, fm):
        batch = fm.shape[0]
        weight_row = norm2_distance(fm[:, :, :-1, :], fm[:, :, 1:, :])
        weight_col = norm2_distance(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weihth_bi = norm2_distance(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        weight_bi = weihth_bi.reshape([batch, -1])
        weight = torch.cat([weight_row, weight_col, weight_bi], dim=1)
        if self.mapping_func is not None:
            weight = self.mapping_func(weight)
        return weight

    def _build_feature_weight_cosine(self, fm, max_tree):
        batch, dim = fm.shape[0], fm.shape[1]
        Tem = 3
        half_width = fm.shape[3] // Tem
        fm_T1_half = fm[:, :, :, :half_width]
        fm_T2_half = fm[:, :, :, half_width:2*half_width]
        fm_T3_half = fm[:, :, :, 2*half_width:3*half_width]
        # T1的weight_row和weight_col
        T1_weight_row = torch.cosine_similarity(fm_T1_half[:, :, :-1, :].reshape(batch, dim, -1),
                                             fm_T1_half[:, :, 1:, :].reshape(batch, dim, -1), dim=1)
        T1_weight_col = torch.cosine_similarity(fm_T1_half[:, :, :, :-1].reshape(batch, dim, -1),
                                             fm_T1_half[:, :, :, 1:].reshape(batch, dim, -1), dim=1)

        # T2的weight_row和weight_col
        T2_weight_row = torch.cosine_similarity(fm_T2_half[:, :, :-1, :].reshape(batch, dim, -1),
                                             fm_T2_half[:, :, 1:, :].reshape(batch, dim, -1), dim=1)
        T2_weight_col = torch.cosine_similarity(fm_T2_half[:, :, :, :-1].reshape(batch, dim, -1),
                                             fm_T2_half[:, :, :, 1:].reshape(batch, dim, -1), dim=1)

        # T3的weight_row和weight_col
        T3_weight_row = torch.cosine_similarity(fm_T3_half[:, :, :-1, :].reshape(batch, dim, -1),
                                             fm_T3_half[:, :, 1:, :].reshape(batch, dim, -1), dim=1)
        T3_weight_col = torch.cosine_similarity(fm_T3_half[:, :, :, :-1].reshape(batch, dim, -1),
                                             fm_T3_half[:, :, :, 1:].reshape(batch, dim, -1), dim=1)

        # 左右对应位置权重
        weight_bi = torch.cosine_similarity(fm_T1_half.reshape(batch, dim, -1),
                                             fm_T3_half.reshape(batch, dim, -1), dim=1)


        weight = torch.cat([T1_weight_row, T1_weight_col, T2_weight_row, T2_weight_col, T3_weight_row, T3_weight_col, weight_bi], dim=1)
        if self.mapping_func is not None:
            if max_tree:
                weight = self.mapping_func(
                    weight)  # cosine similarity needs "-weight" for min tree, "weight" for max tree
            else:
                weight = self.mapping_func(
                    -weight)  # cosine similarity needs "-weight" for min tree, "weight" for max tree
        return weight

    def forward(self, guide_in, max_tree=False):
        with torch.no_grad():
            index = self._build_matrix_index(guide_in)
            if self.distance_func == "Cosine":
                weight = self._build_feature_weight_cosine(guide_in, max_tree)
            else:
                weight = self._build_feature_weight(guide_in)
            tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3])
            # tree = mst(index, weight, guide_in.shape[2])
        return tree


import torch
import torch.nn as nn

class MinimumSpanningMTnDTree(nn.Module):
    def __init__(self, distance_func, mapping_func=None, Tem=2):
        super(MinimumSpanningMTnDTree, self).__init__()
        self.distance_func = distance_func
        self.mapping_func = mapping_func
        self.Tem = Tem

    @staticmethod
    def _build_matrix_index(fm, Tem=2):
        batch, height, width = fm.shape[0], fm.shape[2], fm.shape[3]
        row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
        raw_index = row + col * width
        Tem = 2
        # 动态分割时相
        phase_width = width // Tem
        phase_indices = [raw_index[:, i * phase_width:(i + 1) * phase_width] for i in range(Tem)]

        # 计算每个时相的行索引和列索引
        row_indices = []
        col_indices = []
        for phase_index in phase_indices:
            row_index = torch.stack([phase_index[:-1, :], phase_index[1:, :]], dim=2)
            col_index = torch.stack([phase_index[:, :-1], phase_index[:, 1:]], dim=2)
            row_indices.append(row_index.reshape(1, -1, 2))
            col_indices.append(col_index.reshape(1, -1, 2))

        # 计算跨时相的索引
        cross_indices = []
        for i in range(Tem - 1):
            cross_index = torch.stack([phase_indices[i][:, :], phase_indices[i + 1][:, :]], dim=2)
            cross_indices.append(cross_index.reshape(1, -1, 2))

        # 合并所有索引
        index = torch.cat(row_indices + col_indices + cross_indices, dim=1)
        index = index.expand(batch, -1, -1)

        return index

    def _build_feature_weight(self, fm):
        batch, dim = fm.shape[0], fm.shape[1]
        height, width = fm.shape[2], fm.shape[3]
        phase_width = width // self.Tem

        # 分割特征图
        phase_features = [fm[:, :, :, i * phase_width:(i + 1) * phase_width] for i in range(self.Tem)]

        # 计算每个时相的行权重和列权重
        weights = []
        for phase_feature in phase_features:
            weight_row = torch.cosine_similarity(
                phase_feature[:, :, :-1, :].reshape(batch, dim, -1),
                phase_feature[:, :, 1:, :].reshape(batch, dim, -1),
                dim=1
            )
            weight_col = torch.cosine_similarity(
                phase_feature[:, :, :, :-1].reshape(batch, dim, -1),
                phase_feature[:, :, :, 1:].reshape(batch, dim, -1),
                dim=1
            )
            weights.extend([weight_row, weight_col])

        # 计算跨时相权重
        for i in range(self.Tem - 1):
            weight_cross = torch.cosine_similarity(
                phase_features[i].reshape(batch, dim, -1),
                phase_features[i + 1].reshape(batch, dim, -1),
                dim=1
            )
            weights.append(weight_cross)

        # 合并权重
        weight = torch.cat(weights, dim=1)
        if self.mapping_func is not None:
            weight = self.mapping_func(weight)
        return weight

    def forward(self, guide_in, max_tree=False):
        with torch.no_grad():
            index = self._build_matrix_index(guide_in, self.Tem)
            weight = self._build_feature_weight(guide_in)
            tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3])
        return tree