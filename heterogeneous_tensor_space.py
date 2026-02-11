#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异构张量空间的单步正交嵌入与逆分解实现
Heterogeneous Tensor Space Single-Step Orthogonal Embedding & Inverse Decomposition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, Union, List
import logging
from abc import ABC, abstractmethod
from scipy.linalg import svd, eigh
from scipy.stats import multivariate_normal
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorSpaceTransformer(ABC):
    """张量空间变换器基类"""

    def __init__(self, n: int, K: int, V: int, c: int):
        """
        Args:
            n: 语义向量空间维度
            K: 声学码本深度
            V: 码本特征维度
            c: 统一目标空间维度
        """
        self.n = n
        self.K = K
        self.V = V
        self.c = c

        # 验证维度一致性
        assert c >= n + K * V, f"目标空间维度c={c}必须大于等于n+K*V={n+K*V}"

    @abstractmethod
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """前向变换 T(A, B) = C"""
        pass

    @abstractmethod
    def inverse(self, C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """逆变换 D(C) = (A, B)"""
        pass

    def check_orthogonality(self, C_A: torch.Tensor, C_B: torch.Tensor, eps: float = 1e-6) -> float:
        """检查正交性"""
        inner_product = torch.dot(C_A.flatten(), C_B.flatten())
        norm_A = torch.norm(C_A)
        norm_B = torch.norm(C_B)

        if norm_A == 0 or norm_B == 0:
            return float('inf')

        cosine_sim = inner_product / (norm_A * norm_B)
        return abs(cosine_sim.item())


class ParametricOrthogonalTransformer(TensorSpaceTransformer):
    """参数化正交矩阵法实现"""

    def __init__(self, n: int, K: int, V: int, c: int, method: str = 'cayley'):
        super().__init__(n, K, V, c)
        self.method = method

        # 计算子空间维度
        self.d_A = n  # 语义子空间维度
        self.d_B = K * V  # 声学子空间维度
        assert self.d_A + self.d_B <= c

        # 初始化参数
        self._init_parameters()

        # 构造正交投影矩阵
        self._build_projection_matrices()

    def _init_parameters(self):
        """初始化反对称矩阵参数"""
        if self.method == 'cayley':
            # Cayley变换的反对称矩阵 - 需要是方阵
            self.X_A = nn.Parameter(torch.randn(self.c, self.c) * 0.1)
            self.X_B = nn.Parameter(torch.randn(self.c, self.c) * 0.1)
        elif self.method == 'exponential':
            # 指数映射的反对称矩阵
            self.X_A = nn.Parameter(torch.randn(self.c, self.c) * 0.1)
            self.X_B = nn.Parameter(torch.randn(self.c, self.c) * 0.1)

    def _make_skew_symmetric(self, X: torch.Tensor) -> torch.Tensor:
        """构造反对称矩阵"""
        return (X - X.T) / 2

    def _cayley_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Cayley变换: Q = (I - X)(I + X)^-1"""
        I = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
        skew_X = self._make_skew_symmetric(X)
        Q = torch.inverse(I + skew_X) @ (I - skew_X)
        return Q

    def _exponential_map(self, X: torch.Tensor) -> torch.Tensor:
        """指数映射: Q = Exp(X)"""
        skew_X = self._make_skew_symmetric(X)
        # 使用矩阵指数的泰勒级数近似
        Q = torch.matrix_exp(skew_X)
        return Q

    def _build_projection_matrices(self):
        """构建正交投影矩阵"""
        if self.method == 'cayley':
            Q_A = self._cayley_transform(self.X_A)
            Q_B = self._cayley_transform(self.X_B)
        elif self.method == 'exponential':
            Q_A = self._exponential_map(self.X_A)
            Q_B = self._exponential_map(self.X_B)

        # 截取前d_A和d_B列作为子空间基
        self.Q_A = Q_A[:, :self.d_A]  # (c, d_A)
        self.Q_B = Q_B[:, :self.d_B]  # (c, d_B)

        # 正交化子空间
        self.Q_B = self._gram_schmidt(self.Q_B, self.Q_A)

    def _gram_schmidt(self, Q_new: torch.Tensor, Q_ref: torch.Tensor) -> torch.Tensor:
        """Gram-Schmidt正交化"""
        # 投影到参考空间的正交补
        projection = Q_ref @ (Q_ref.T @ Q_new)
        orthogonal = Q_new - projection

        # 再次正交化以确保数值稳定性
        for i in range(orthogonal.shape[1]):
            for j in range(i):
                orthogonal[:, i] -= (orthogonal[:, j].T @ orthogonal[:, i]) * orthogonal[:, j]
            orthogonal[:, i] /= torch.norm(orthogonal[:, i]) + 1e-8

        return orthogonal

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """前向变换 C = phi(A) + psi(B)"""
        batch_size = A.shape[0]

        # 输入形状验证
        assert A.shape[1] == self.n, f"A should have {self.n} features, got {A.shape[1]}"
        assert B.shape == (batch_size, self.K, self.V), f"B shape should be ({batch_size}, {self.K}, {self.V}), got {B.shape}"

        # 展平声学张量
        B_flat = B.view(batch_size, self.K * self.V)  # (batch_size, d_B)

        # 映射到统一空间 - 使用伪逆处理非方阵情况
        C_A = A @ torch.pinverse(self.Q_A[:self.n, :]) @ self.Q_A.T  # (batch_size, c)
        C_B = B_flat @ torch.pinverse(self.Q_B[:self.d_B, :]) @ self.Q_B.T  # (batch_size, c)

        C = C_A + C_B  # (batch_size, c)
        return C

    def inverse(self, C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """逆变换 D(C) = (A, B)"""
        batch_size = C.shape[0]

        # 正交投影
        C_A = C @ self.Q_A @ self.Q_A.T  # 投影到语义子空间
        C_B = C @ self.Q_B @ self.Q_B.T  # 投影到声学子空间

        # 逆映射 - 从投影空间恢复原始空间
        A = C_A @ self.Q_A @ torch.pinverse(self.Q_A[:self.n, :].T)  # (batch_size, c) -> (batch_size, n)
        B_flat = C_B @ self.Q_B @ torch.pinverse(self.Q_B[:self.d_B, :].T)  # (batch_size, c) -> (batch_size, d_B)
        B = B_flat.view(batch_size, self.K, self.V)  # (batch_size, d_B) -> (batch_size, K, V)

        return A, B


class CCATransformer(TensorSpaceTransformer):
    """统计对齐法(CCA)实现"""

    def __init__(self, n: int, K: int, V: int, c: int, regularization: float = 1e-6):
        super().__init__(n, K, V, c)
        self.regularization = regularization

        # 计算子空间维度
        self.d_A = n
        self.d_B = K * V

        # CCA变换矩阵（需要在数据上拟合）
        self.W_A = None
        self.W_B = None
        self.is_fitted = False

    def fit(self, A_samples: torch.Tensor, B_samples: torch.Tensor):
        """在样本数据上拟合CCA变换"""
        assert A_samples.shape[1] == self.d_A
        assert B_samples.shape[1] == self.d_B
        assert A_samples.shape[0] == B_samples.shape[0]

        # 转换为numpy
        A_np = A_samples.detach().cpu().numpy()
        B_np = B_samples.detach().cpu().numpy()

        # 计算协方差矩阵
        n_samples = A_samples.shape[0]
        A_centered = A_np - np.mean(A_np, axis=0, keepdims=True)
        B_centered = B_np - np.mean(B_np, axis=0, keepdims=True)

        Sigma_AA = (A_centered.T @ A_centered) / (n_samples - 1) + self.regularization * np.eye(self.d_A)
        Sigma_BB = (B_centered.T @ B_centered) / (n_samples - 1) + self.regularization * np.eye(self.d_B)
        Sigma_AB = (A_centered.T @ B_centered) / (n_samples - 1)

        # 广义特征值问题
        # Sigma_AB @ Sigma_BB^-1 @ Sigma_BA @ w_A = lambda^2 @ Sigma_AA @ w_A
        Sigma_BB_inv = np.linalg.inv(Sigma_BB)
        M = Sigma_AB @ Sigma_BB_inv @ Sigma_AB.T

        # 求解特征值问题
        eigenvalues, eigenvectors = eigh(M, Sigma_AA)

        # 按特征值降序排序
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.W_A_np = eigenvectors[:, idx]

        # 计算W_B
        self.W_B_np = Sigma_BB_inv @ Sigma_AB.T @ self.W_A_np

        # 归一化
        for i in range(min(self.d_A, self.d_B)):
            norm_A = np.sqrt(self.W_A_np[:, i].T @ Sigma_AA @ self.W_A_np[:, i])
            norm_B = np.sqrt(self.W_B_np[:, i].T @ Sigma_BB @ self.W_B_np[:, i])
            self.W_A_np[:, i] /= norm_A
            self.W_B_np[:, i] /= norm_B

        # 转换为torch tensors
        self.W_A = torch.tensor(self.W_A_np, dtype=torch.float32)
        self.W_B = torch.tensor(self.W_B_np, dtype=torch.float32)

        # 构建扩展到c维的投影矩阵
        self._build_expansion_matrices()

        self.is_fitted = True
        logger.info(f"CCA拟合完成，最大特征值: {self.eigenvalues[0]:.4f}")

    def _build_expansion_matrices(self):
        """构建扩展到c维的投影矩阵"""
        # 使用前min(d_A, d_B)个典型变量
        n_components = min(self.d_A, self.d_B, (self.c) // 2)

        # 扩展到c维的基矩阵
        self.expansion_A = torch.zeros(self.d_A, n_components)
        self.expansion_B = torch.zeros(self.d_B, n_components)

        self.expansion_A[:, :n_components] = self.W_A[:, :n_components]
        self.expansion_B[:, :n_components] = self.W_B[:, :n_components]

        # 如果还有剩余维度，用正交补空间填充
        remaining_dims = self.c - 2 * n_components
        if remaining_dims > 0:
            self.ortho_complement = torch.randn(remaining_dims, self.c)
            self.ortho_complement, _ = torch.qr(self.ortho_complement.T)
            self.ortho_complement = self.ortho_complement.T

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """前向变换"""
        if not self.is_fitted:
            raise RuntimeError("CCA模型未拟合，请先调用fit方法")

        # 输入形状验证
        batch_size = A.shape[0]
        assert A.shape[1] == self.n
        assert B.shape == (batch_size, self.K, self.V)

        # 展平声学张量
        B_flat = B.view(batch_size, self.K * self.V)

        # CCA投影
        A_proj = A @ self.expansion_A  # (1, n_components)
        B_proj = B_flat @ self.expansion_B  # (1, n_components)

        # 合并到统一空间
        if hasattr(self, 'ortho_complement'):
            C_combined = torch.cat([A_proj, B_proj, torch.zeros(1, self.ortho_complement.shape[0])], dim=-1)
        else:
            C_combined = torch.cat([A_proj, B_proj], dim=-1)

        # 如果维度不足，填充零
        if C_combined.shape[1] < self.c:
            padding = torch.zeros(1, self.c - C_combined.shape[1])
            C_combined = torch.cat([C_combined, padding], dim=-1)

        return C_combined

    def inverse(self, C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """逆变换"""
        batch_size = C.shape[0]

        if not self.is_fitted:
            raise RuntimeError("CCA模型未拟合，请先调用fit方法")

        # 提取对应的前components维度
        n_components = min(self.d_A, self.d_B, (self.c) // 2)

        C_A = C[:, :n_components]
        C_B = C[:, n_components:2*n_components]

        # 逆投影A: (1, n_components) -> (1, d_A)
        A = C_A @ torch.pinverse(self.expansion_A[:n_components, :].T)

        # 逆投影B: 由于B的原始维度大于n_components，完全重构是不可能的
        # 我们重构到压缩表示，然后重建损失的维度
        if self.d_B > n_components:
            # 重构到压缩空间，然后用零填充
            B_flat_compressed = C_B @ torch.pinverse(self.expansion_B[:n_components, :].T)  # (1, n_components)
            B_flat = torch.zeros(1, self.d_B)
            B_flat[:, :n_components] = B_flat_compressed
        else:
            # 如果原始维度小于等于components，可以直接重构
            B_flat = C_B @ torch.pinverse(self.expansion_B[:n_components, :].T)

        B = B_flat.view(batch_size, self.K, self.V)

        return A, B


class ContrastiveTransformer(TensorSpaceTransformer):
    """对比学习对齐法实现"""

    def __init__(self, n: int, K: int, V: int, c: int, temperature: float = 0.07, hidden_dim: int = 256):
        super().__init__(n, K, V, c)
        self.temperature = temperature
        self.hidden_dim = hidden_dim

        # 特征提取器
        self.semantic_encoder = nn.Sequential(
            nn.Linear(n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, c // 2)
        )

        self.acoustic_encoder = nn.Sequential(
            nn.Linear(K * V, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, c // 2)
        )

        # 对比学习投影头
        self.semantic_proj = nn.Sequential(
            nn.Linear(c // 2, c // 4),
            nn.ReLU(),
            nn.Linear(c // 4, c // 4)
        )

        self.acoustic_proj = nn.Sequential(
            nn.Linear(c // 2, c // 4),
            nn.ReLU(),
            nn.Linear(c // 4, c // 4)
        )

        # 正交化层
        self.orthogonalizer = OrthogonalLayer(c // 4)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """前向变换"""
        batch_size = A.shape[0]

        # 输入形状验证
        assert A.shape[1] == self.n
        assert B.shape == (batch_size, self.K, self.V)

        # 展平声学张量
        B_flat = B.view(batch_size, self.K * self.V)

        # 特征提取
        semantic_features = self.semantic_encoder(A)  # (batch_size, c//2)
        acoustic_features = self.acoustic_encoder(B_flat)  # (batch_size, c//2)

        # 投影到对比空间
        semantic_proj = self.semantic_proj(semantic_features)  # (batch_size, c//4)
        acoustic_proj = self.acoustic_proj(acoustic_features)  # (batch_size, c//4)

        # 正交化（逐样本处理）
        semantic_ortho_list = []
        acoustic_ortho_list = []

        for i in range(batch_size):
            sem_ortho, ac_ortho = self.orthogonalizer(
                semantic_proj[i:i+1], acoustic_proj[i:i+1]
            )
            semantic_ortho_list.append(sem_ortho)
            acoustic_ortho_list.append(ac_ortho)

        semantic_ortho = torch.cat(semantic_ortho_list, dim=0)  # (batch_size, c//4)
        acoustic_ortho = torch.cat(acoustic_ortho_list, dim=0)  # (batch_size, c//4)

        # 合并到统一空间
        C = torch.cat([semantic_ortho, acoustic_ortho], dim=-1)  # (batch_size, c//2)

        # 填充到完整c维
        if C.shape[1] < self.c:
            padding = torch.zeros(batch_size, self.c - C.shape[1])
            C = torch.cat([C, padding], dim=-1)

        return C

    def inverse(self, C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """逆变换（使用decoder）"""
        batch_size = C.shape[0]

        if not hasattr(self, 'semantic_decoder'):
            # 在训练后构建逆映射
            self.semantic_decoder = nn.Sequential(
                nn.Linear(self.c // 4, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.n)
            )

            self.acoustic_decoder = nn.Sequential(
                nn.Linear(self.c // 4, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.K * self.V)
            )

        # 分离语义和声学部分
        # C的前半部分是有效数据，后半部分是填充的零
        effective_dim = self.c // 2  # 有效数据的维度
        C_effective = C[:, :effective_dim]  # 取前c//2维

        # 在有效数据中分离语义和声学部分
        split_point = effective_dim // 2  # c//4
        C_A = C_effective[:, :split_point]  # (batch_size, c//4)
        C_B = C_effective[:, split_point:]  # (batch_size, c//4)

        # 逆映射
        A = self.semantic_decoder(C_A)
        B_flat = self.acoustic_decoder(C_B)
        B = B_flat.view(batch_size, self.K, self.V)

        return A, B

    def contrastive_loss(self, semantic_features: torch.Tensor, acoustic_features: torch.Tensor) -> torch.Tensor:
        """计算对比损失"""
        # 归一化
        semantic_norm = F.normalize(semantic_features, dim=-1)
        acoustic_norm = F.normalize(acoustic_features, dim=-1)

        # 计算相似度矩阵
        logits = torch.matmul(semantic_norm, acoustic_norm.T) / self.temperature

        # 标签是正样本对角线
        batch_size = semantic_features.shape[0]
        labels = torch.arange(batch_size, device=semantic_features.device)

        # 计算双向损失
        loss_sem_to_acoustic = F.cross_entropy(logits, labels)
        loss_acoustic_to_sem = F.cross_entropy(logits.T, labels)

        return (loss_sem_to_acoustic + loss_acoustic_to_sem) / 2


class OrthogonalLayer(nn.Module):
    """正交化层"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """使两个向量正交"""
        # Gram-Schmidt正交化
        x_norm = F.normalize(x, dim=-1)
        y_proj_on_x = torch.sum(y * x_norm, dim=-1, keepdim=True) * x_norm
        y_ortho = y - y_proj_on_x
        y_ortho_norm = F.normalize(y_ortho, dim=-1)

        return x_norm, y_ortho_norm


class HeterogeneousTensorSystem:
    """异构张量空间完整系统"""

    def __init__(self, n: int, K: int, V: int, c: int, method: str = 'parametric'):
        """
        Args:
            n: 语义向量空间维度
            K: 声学码本深度
            V: 码本特征维度
            c: 统一目标空间维度
            method: 变换方法 ('parametric', 'cca', 'contrastive')
        """
        self.n = n
        self.K = K
        self.V = V
        self.c = c

        if method == 'parametric':
            self.transformer = ParametricOrthogonalTransformer(n, K, V, c)
        elif method == 'cca':
            self.transformer = CCATransformer(n, K, V, c)
        elif method == 'contrastive':
            self.transformer = ContrastiveTransformer(n, K, V, c)
        else:
            raise ValueError(f"未知方法: {method}")

        self.metrics_history = []

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """前向变换"""
        return self.transformer.forward(A, B)

    def inverse(self, C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """逆变换"""
        return self.transformer.inverse(C)

    def evaluate(self, A: torch.Tensor, B: torch.Tensor) -> Dict[str, float]:
        """评估变换性能"""
        C = self.forward(A, B)
        A_recon, B_recon = self.inverse(C)

        # 计算重构误差
        recon_error_A = F.mse_loss(A_recon, A).item()
        recon_error_B = F.mse_loss(B_recon, B).item()
        recon_error_total = recon_error_A + recon_error_B

        # 计算正交性（如果支持）
        orthogonality = 0.0
        if hasattr(self.transformer, 'forward'):
            # 重新计算以获取中间结果
            if hasattr(self.transformer, 'semantic_encoder'):  # ContrastiveTransformer
                B_flat = B.view(1, self.K * self.V)
                semantic_features = self.transformer.semantic_encoder(A)
                acoustic_features = self.transformer.acoustic_encoder(B_flat)
                semantic_proj = self.transformer.semantic_proj(semantic_features)
                acoustic_proj = self.transformer.acoustic_proj(acoustic_features)
                orthogonality = self.transformer.check_orthogonality(semantic_proj, acoustic_proj)

        metrics = {
            'recon_error_A': recon_error_A,
            'recon_error_B': recon_error_B,
            'recon_error_total': recon_error_total,
            'orthogonality': orthogonality
        }

        self.metrics_history.append(metrics)
        return metrics

    def train_step(self, A_batch: torch.Tensor, B_batch: torch.Tensor, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """训练步骤（用于对比学习方法）"""
        optimizer.zero_grad()

        # 前向变换
        C_batch = self.forward(A_batch, B_batch)

        # 重构损失
        A_recon, B_recon = self.inverse(C_batch)
        recon_loss = F.mse_loss(A_recon, A_batch) + F.mse_loss(B_recon, B_batch)

        total_loss = recon_loss

        # 如果是对比学习，添加对比损失
        if hasattr(self.transformer, 'contrastive_loss'):
            B_flat_batch = B_batch.view(B_batch.shape[0], self.K * self.V)
            semantic_features = self.transformer.semantic_encoder(A_batch)
            acoustic_features = self.transformer.acoustic_encoder(B_flat_batch)
            semantic_proj = self.transformer.semantic_proj(semantic_features)
            acoustic_proj = self.transformer.acoustic_proj(acoustic_features)
            contrastive_loss = self.transformer.contrastive_loss(semantic_proj, acoustic_proj)
            total_loss = recon_loss + 0.1 * contrastive_loss

        # 反向传播
        total_loss.backward()
        optimizer.step()

        # 重新正交化（如果是参数化方法）
        if hasattr(self.transformer, '_build_projection_matrices'):
            self.transformer._build_projection_matrices()

        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'contrastive_loss': contrastive_loss.item() if 'contrastive_loss' in locals() else 0.0
        }


def test_transformer():
    """测试异构张量变换器"""
    print("=" * 60)
    print("异构张量空间正交嵌入测试")
    print("=" * 60)

    # 测试参数
    n, K, V, c = 128, 4, 64, 512
    batch_size = 8

    print(f"空间维度: A∈R^(1×{n}), B∈R^(1×{K}×{V}), C∈R^(1×{c})")

    # 测试三种方法
    methods = ['parametric', 'cca', 'contrastive']

    for method in methods:
        print(f"\n{'='*40}")
        print(f"测试方法: {method.upper()}")
        print(f"{'='*40}")

        # 创建系统
        system = HeterogeneousTensorSystem(n, K, V, c, method)

        # 生成测试数据
        A = torch.randn(1, n)
        B = torch.randn(1, K, V)

        # 测试前向变换
        print("执行前向变换...")
        C = system.forward(A, B)
        print(f"✓ 前向变换成功: A{A.shape}, B{B.shape} -> C{C.shape}")

        # 测试逆变换
        print("执行逆变换...")
        A_recon, B_recon = system.inverse(C)
        print(f"✓ 逆变换成功: C{C.shape} -> A{A_recon.shape}, B{B_recon.shape}")

        # 计算性能指标
        metrics = system.evaluate(A, B)
        print(f"重构误差: A={metrics['recon_error_A']:.6f}, B={metrics['recon_error_B']:.6f}")
        print(f"正交性指标: {metrics['orthogonality']:.6f}")

        # CCA方法需要先拟合
        if method == 'cca':
            print("CCA方法: 生成更多数据进行拟合...")
            A_samples = torch.randn(100, n)
            B_samples = torch.randn(100, K, V).view(100, K * V)
            system.transformer.fit(A_samples, B_samples)

            # 重新测试
            C = system.forward(A, B)
            A_recon, B_recon = system.inverse(C)
            metrics = system.evaluate(A, B)
            print(f"CCA拟合后重构误差: {metrics['recon_error_total']:.6f}")

        # 对比学习方法需要训练
        if method == 'contrastive':
            print("对比学习方法: 执行训练步骤...")
            optimizer = torch.optim.Adam(system.transformer.parameters(), lr=1e-3)

            A_batch = torch.randn(batch_size, n)
            B_batch = torch.randn(batch_size, K, V)

            train_metrics = system.train_step(A_batch, B_batch, optimizer)
            print(f"训练损失: {train_metrics}")

        print(f"✓ {method.upper()}方法测试完成")


if __name__ == "__main__":
    # 运行测试
    test_transformer()

    print("\n" + "=" * 60)
    print("异构张量空间正交嵌入系统完成！")
    print("=" * 60)