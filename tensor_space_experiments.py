#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异构张量空间数值验证实验
Heterogeneous Tensor Space Numerical Validation Experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from heterogeneous_tensor_space import (
    HeterogeneousTensorSystem,
    ParametricOrthogonalTransformer,
    CCATransformer,
    ContrastiveTransformer
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class TensorSpaceExperimentSuite:
    """异构张量空间实验套件"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.experiment_results = {}
        self.figures = {}

    def generate_synthetic_data(self, n_samples: int, n: int, K: int, V: int,
                               correlation_strength: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成合成数据"""
        print(f"生成合成数据: {n_samples}个样本, A∈R^{n}, B∈R^{K}×{V}")

        # 生成基础语义向量
        A_base = torch.randn(n_samples, n)

        # 生成声学特征（与A有一定相关性）
        B_flat = torch.randn(n_samples, K * V)

        # 添加相关性
        if correlation_strength > 0:
            # 使用A的部分维度影响B
            n_shared = min(n // 2, K * V // 2)
            shared_component = A_base[:, :n_shared] @ torch.randn(n_shared, n_shared)
            B_flat[:, :n_shared] = correlation_strength * shared_component + \
                                   (1 - correlation_strength) * B_flat[:, :n_shared]

        B = B_flat.view(n_samples, K, V)

        return A_base, B

    def test_orthogonality_preservation(self, system: HeterogeneousTensorSystem,
                                      n_tests: int = 100) -> Dict[str, float]:
        """测试正交性保持"""
        print("测试正交性保持...")

        orthogonality_scores = []

        for _ in range(n_tests):
            A = torch.randn(1, system.n)
            B = torch.randn(1, system.K, system.V)

            C = system.forward(A, B)
            A_recon, B_recon = system.inverse(C)

            # 计算重构的特征
            if isinstance(system.transformer, ContrastiveTransformer):
                B_flat = B.view(1, system.K * system.V)
                semantic_features = system.transformer.semantic_encoder(A)
                acoustic_features = system.transformer.acoustic_encoder(B_flat)
                semantic_proj = system.transformer.semantic_proj(semantic_features)
                acoustic_proj = system.transformer.acoustic_proj(acoustic_features)

                ortho_score = system.transformer.check_orthogonality(semantic_proj, acoustic_proj)
                orthogonality_scores.append(ortho_score)
            else:
                # 对于其他方法，计算原始特征的重构误差间接评估正交性
                recon_error = F.mse_loss(A_recon, A) + F.mse_loss(B_recon, B)
                orthogonality_scores.append(1.0 / (1.0 + recon_error.item()))

        return {
            'mean_orthogonality': np.mean(orthogonality_scores),
            'std_orthogonality': np.std(orthogonality_scores),
            'min_orthogonality': np.min(orthogonality_scores),
            'max_orthogonality': np.max(orthogonality_scores)
        }

    def test_reconstruction_accuracy(self, system: HeterogeneousTensorSystem,
                                   test_samples: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """测试重构准确度"""
        print("测试重构准确度...")

        A_test, B_test = test_samples
        n_samples = A_test.shape[0]

        recon_errors_A = []
        recon_errors_B = []
        relative_errors_A = []
        relative_errors_B = []

        for i in range(n_samples):
            A = A_test[i:i+1]
            B = B_test[i:i+1]

            C = system.forward(A, B)
            A_recon, B_recon = system.inverse(C)

            # 绝对重构误差
            error_A = F.mse_loss(A_recon, A).item()
            error_B = F.mse_loss(B_recon, B).item()
            recon_errors_A.append(error_A)
            recon_errors_B.append(error_B)

            # 相对重构误差
            rel_error_A = error_A / (torch.norm(A).item()**2 + 1e-8)
            rel_error_B = error_B / (torch.norm(B).item()**2 + 1e-8)
            relative_errors_A.append(rel_error_A)
            relative_errors_B.append(rel_error_B)

        return {
            'mean_recon_error_A': np.mean(recon_errors_A),
            'std_recon_error_A': np.std(recon_errors_A),
            'mean_recon_error_B': np.mean(recon_errors_B),
            'std_recon_error_B': np.std(recon_errors_B),
            'mean_relative_error_A': np.mean(relative_errors_A),
            'mean_relative_error_B': np.mean(relative_errors_B)
        }

    def test_convergence(self, method_name: str, n: int, K: int, V: int, c: int,
                        n_epochs: int = 100) -> Dict[str, List[float]]:
        """测试收敛性"""
        print(f"测试 {method_name} 方法收敛性...")

        # 创建系统
        system = HeterogeneousTensorSystem(n, K, V, c, method_name)

        convergence_history = {
            'loss': [],
            'recon_error_A': [],
            'recon_error_B': [],
            'orthogonality': []
        }

        if method_name == 'cca':
            # CCA方法需要拟合
            A_train, B_train = self.generate_synthetic_data(200, n, K, V)
            B_train_flat = B_train.view(200, K * V)
            system.transformer.fit(A_train, B_train_flat)

            # 验证收敛
            A_test, B_test = self.generate_synthetic_data(50, n, K, V)
            for i in range(min(n_epochs, 20)):
                metrics = self.test_reconstruction_accuracy(system, (A_test, B_test))
                ortho_metrics = self.test_orthogonality_preservation(system, 20)

                convergence_history['loss'].append(metrics['mean_recon_error_A'] + metrics['mean_recon_error_B'])
                convergence_history['recon_error_A'].append(metrics['mean_recon_error_A'])
                convergence_history['recon_error_B'].append(metrics['mean_recon_error_B'])
                convergence_history['orthogonality'].append(ortho_metrics['mean_orthogonality'])

        elif method_name == 'contrastive':
            # 对比学习需要训练
            # 获取ContrastiveTransformer的所有nn.Module参数
            modules_to_optimize = []
            for module_name in ['semantic_encoder', 'acoustic_encoder', 'semantic_proj', 'acoustic_proj', 'orthogonalizer']:
                if hasattr(system.transformer, module_name):
                    module = getattr(system.transformer, module_name)
                    if isinstance(module, nn.Module):
                        modules_to_optimize.extend(list(module.parameters()))

            optimizer = torch.optim.Adam(modules_to_optimize, lr=1e-3)
            batch_size = 16

            for epoch in range(n_epochs):
                # 生成批次数据
                A_batch = torch.randn(batch_size, n)
                B_batch = torch.randn(batch_size, K, V)

                # 训练步骤
                train_metrics = system.train_step(A_batch, B_batch, optimizer)

                # 记录收敛历史
                convergence_history['loss'].append(train_metrics['total_loss'])
                convergence_history['recon_error_A'].append(train_metrics['recon_loss'] / 2)
                convergence_history['recon_error_B'].append(train_metrics['recon_loss'] / 2)

                # 计算正交性
                if epoch % 10 == 0:
                    ortho_metrics = self.test_orthogonality_preservation(system, 10)
                    convergence_history['orthogonality'].append(ortho_metrics['mean_orthogonality'])
                else:
                    convergence_history['orthogonality'].append(convergence_history['orthogonality'][-1] if convergence_history['orthogonality'] else 0)

                if epoch % 20 == 0:
                    print(f"  Epoch {epoch}: Loss={train_metrics['total_loss']:.6f}")

        elif method_name == 'parametric':
            # 参数化方法测试稳定性
            for i in range(n_epochs):
                A_test = torch.randn(1, n)
                B_test = torch.randn(1, K, V)

                metrics = self.test_reconstruction_accuracy(system, (A_test, B_test))
                ortho_metrics = self.test_orthogonality_preservation(system, 5)

                convergence_history['loss'].append(metrics['mean_recon_error_A'] + metrics['mean_recon_error_B'])
                convergence_history['recon_error_A'].append(metrics['mean_recon_error_A'])
                convergence_history['recon_error_B'].append(metrics['mean_recon_error_B'])
                convergence_history['orthogonality'].append(ortho_metrics['mean_orthogonality'])

        return convergence_history

    def test_robustness(self, system: HeterogeneousTensorSystem,
                       A_test: torch.Tensor, B_test: torch.Tensor,
                       noise_levels: List[float] = None) -> Dict[str, List[float]]:
        """测试鲁棒性"""
        print("测试鲁棒性...")

        if noise_levels is None:
            noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

        robustness_results = {
            'noise_levels': noise_levels,
            'recon_errors': [],
            'relative_errors': []
        }

        original_A = A_test.clone()
        original_B = B_test.clone()

        for noise_level in noise_levels:
            total_error = 0.0
            relative_error = 0.0
            n_trials = 10

            for _ in range(n_trials):
                # 添加噪声
                A_noisy = original_A + noise_level * torch.randn_like(original_A)
                B_noisy = original_B + noise_level * torch.randn_like(original_B)

                # 变换和重构
                C = system.forward(A_noisy, B_noisy)
                A_recon, B_recon = system.inverse(C)

                # 计算误差相对于原始信号
                error = F.mse_loss(A_recon, original_A) + F.mse_loss(B_recon, original_B)
                rel_error = error / (torch.norm(original_A)**2 + torch.norm(original_B)**2 + 1e-8)

                total_error += error.item()
                relative_error += rel_error.item()

            robustness_results['recon_errors'].append(total_error / n_trials)
            robustness_results['relative_errors'].append(relative_error / n_trials)

        return robustness_results

    def compare_methods(self, n: int, K: int, V: int, c: int) -> Dict[str, Any]:
        """比较三种方法"""
        print("="*60)
        print("方法性能比较")
        print("="*60)

        methods = ['parametric', 'cca', 'contrastive']
        comparison_results = {}

        # 生成测试数据
        A_test, B_test = self.generate_synthetic_data(50, n, K, V, correlation_strength=0.3)

        for method in methods:
            print(f"\n测试 {method.upper()} 方法...")
            start_time = time.time()

            # 创建系统
            system = HeterogeneousTensorSystem(n, K, V, c, method)

            # 特殊处理CCA方法
            if method == 'cca':
                A_train, B_train = self.generate_synthetic_data(100, n, K, V)
                B_train_flat = B_train.view(100, K * V)
                system.transformer.fit(A_train, B_train_flat)

            # 特殊处理Contrastive方法
            elif method == 'contrastive':
                # 获取ContrastiveTransformer的所有nn.Module参数
                modules_to_optimize = []
                for module_name in ['semantic_encoder', 'acoustic_encoder', 'semantic_proj', 'acoustic_proj', 'orthogonalizer']:
                    if hasattr(system.transformer, module_name):
                        module = getattr(system.transformer, module_name)
                        if isinstance(module, nn.Module):
                            modules_to_optimize.extend(list(module.parameters()))

                optimizer = torch.optim.Adam(modules_to_optimize, lr=1e-3)
                for epoch in range(50):
                    A_batch = torch.randn(16, n)
                    B_batch = torch.randn(16, K, V)
                    system.train_step(A_batch, B_batch, optimizer)

            # 测试性能
            recon_metrics = self.test_reconstruction_accuracy(system, (A_test, B_test))
            ortho_metrics = self.test_orthogonality_preservation(system, 30)
            robustness_metrics = self.test_robustness(system, A_test[0:1], B_test[0:1])

            computation_time = time.time() - start_time

            comparison_results[method] = {
                'reconstruction_error': recon_metrics['mean_recon_error_A'] + recon_metrics['mean_recon_error_B'],
                'orthogonality_score': ortho_metrics['mean_orthogonality'],
                'robustness_score': 1.0 / (1.0 + np.mean(robustness_metrics['relative_errors'][1:])),  # 除了无噪声情况
                'computation_time': computation_time,
                'detailed_metrics': {
                    'reconstruction': recon_metrics,
                    'orthogonality': ortho_metrics,
                    'robustness': robustness_metrics
                }
            }

            print(f"  重构误差: {comparison_results[method]['reconstruction_error']:.6f}")
            print(f"  正交性分数: {comparison_results[method]['orthogonality_score']:.6f}")
            print(f"  鲁棒性分数: {comparison_results[method]['robustness_score']:.6f}")
            print(f"  计算时间: {computation_time:.3f}秒")

        self.experiment_results['method_comparison'] = comparison_results
        return comparison_results

    def visualize_results(self):
        """可视化实验结果"""
        print("生成可视化结果...")

        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(16, 12))

        # 1. 方法比较雷达图
        ax1 = plt.subplot(2, 3, 1, projection='polar')
        if 'method_comparison' in self.experiment_results:
            methods = list(self.experiment_results['method_comparison'].keys())
            metrics = ['reconstruction_error', 'orthogonality_score', 'robustness_score']

            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # 闭合

            for method in methods:
                results = self.experiment_results['method_comparison'][method]
                values = [results[metric] for metric in metrics]
                # 标准化到0-1范围
                values = [1.0 / (1.0 + v) if metric == 'reconstruction_error' else v for metric, v in zip(metrics, values)]
                values += values[:1]  # 闭合

                ax1.plot(angles, values, 'o-', linewidth=2, label=method.upper())
                ax1.fill(angles, values, alpha=0.25)

            ax1.set_xticks(angles[:-1])
            ax1.set_xticklabels(['重构准确度', '正交性', '鲁棒性'])
            ax1.set_ylim(0, 1)
            ax1.set_title('方法性能比较', pad=20)
            ax1.legend()

        # 2. 收敛性曲线
        ax2 = plt.subplot(2, 3, 2)
        if 'convergence' in self.experiment_results:
            for method, history in self.experiment_results['convergence'].items():
                epochs = range(len(history['loss']))
                ax2.plot(epochs, history['loss'], label=method.upper(), linewidth=2)
            ax2.set_xlabel('训练轮次')
            ax2.set_ylabel('损失值')
            ax2.set_title('收敛性比较')
            ax2.legend()
            ax2.set_yscale('log')

        # 3. 鲁棒性测试
        ax3 = plt.subplot(2, 3, 3)
        if 'method_comparison' in self.experiment_results:
            for method in self.experiment_results['method_comparison']:
                robustness = self.experiment_results['method_comparison'][method]['detailed_metrics']['robustness']
                ax3.plot(robustness['noise_levels'], robustness['relative_errors'],
                        'o-', label=method.upper(), linewidth=2)
            ax3.set_xlabel('噪声水平')
            ax3.set_ylabel('相对重构误差')
            ax3.set_title('鲁棒性测试')
            ax3.legend()
            ax3.set_yscale('log')

        # 4. 正交性保持
        ax4 = plt.subplot(2, 3, 4)
        if 'method_comparison' in self.experiment_results:
            methods = list(self.experiment_results['method_comparison'].keys())
            ortho_scores = [self.experiment_results['method_comparison'][m]['orthogonality_score'] for m in methods]
            bars = ax4.bar(methods, ortho_scores, color=['skyblue', 'lightgreen', 'salmon'])
            ax4.set_ylabel('正交性分数')
            ax4.set_title('正交性保持比较')
            ax4.set_ylim(0, 1)

            # 添加数值标签
            for bar, score in zip(bars, ortho_scores):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')

        # 5. 重构误差分布
        ax5 = plt.subplot(2, 3, 5)
        if 'method_comparison' in self.experiment_results:
            recon_errors = []
            method_labels = []
            for method in self.experiment_results['method_comparison']:
                recon_errors.append(self.experiment_results['method_comparison'][method]['reconstruction_error'])
                method_labels.append(method.upper())

            bars = ax5.bar(method_labels, recon_errors, color=['skyblue', 'lightgreen', 'salmon'])
            ax5.set_ylabel('重构误差')
            ax5.set_title('重构误差比较')
            ax5.set_yscale('log')

            # 添加数值标签
            for bar, error in zip(bars, recon_errors):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                        f'{error:.2e}', ha='center', va='bottom')

        # 6. 计算效率比较
        ax6 = plt.subplot(2, 3, 6)
        if 'method_comparison' in self.experiment_results:
            methods = list(self.experiment_results['method_comparison'].keys())
            times = [self.experiment_results['method_comparison'][m]['computation_time'] for m in methods]
            bars = ax6.bar(methods, times, color=['skyblue', 'lightgreen', 'salmon'])
            ax6.set_ylabel('计算时间 (秒)')
            ax6.set_title('计算效率比较')

            # 添加数值标签
            for bar, time in zip(bars, times):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{time:.3f}s', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('tensor_space_experiments_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def run_full_experiment_suite(self):
        """运行完整实验套件"""
        print("=" * 80)
        print("异构张量空间数值验证实验套件")
        print("=" * 80)

        # 实验参数
        experiment_configs = [
            {'n': 64, 'K': 4, 'V': 32, 'c': 256, 'name': '小规模'},
            {'n': 128, 'K': 8, 'V': 64, 'c': 512, 'name': '中规模'},
            {'n': 256, 'K': 16, 'V': 128, 'c': 1024, 'name': '大规模'}
        ]

        for config in experiment_configs:
            print(f"\n{'='*60}")
            print(f"配置: {config['name']} (n={config['n']}, K={config['K']}, V={config['V']}, c={config['c']})")
            print(f"{'='*60}")

            # 1. 方法比较
            comparison_results = self.compare_methods(config['n'], config['K'], config['V'], config['c'])

            # 2. 收敛性测试
            print(f"\n收敛性测试...")
            convergence_results = {}
            for method in ['parametric', 'cca', 'contrastive']:
                convergence_results[method] = self.test_convergence(
                    method, config['n'], config['K'], config['V'], config['c']
                )
            self.experiment_results[f'convergence_{config["name"]}'] = convergence_results

        # 生成可视化
        print(f"\n{'='*60}")
        print("生成实验结果可视化...")
        print(f"{'='*60}")
        fig = self.visualize_results()
        self.figures['main_results'] = fig

        # 打印总结
        print(f"\n{'='*60}")
        print("实验总结")
        print(f"{'='*60}")

        for config in experiment_configs:
            if f'convergence_{config["name"]}' in self.experiment_results:
                print(f"\n{config['name']}配置:")
                if 'method_comparison' in self.experiment_results:
                    best_method = min(self.experiment_results['method_comparison'].items(),
                                   key=lambda x: x[1]['reconstruction_error'])
                    print(f"  最佳重构方法: {best_method[0].upper()} (误差: {best_method[1]['reconstruction_error']:.6f})")

                    best_ortho = max(self.experiment_results['method_comparison'].items(),
                                  key=lambda x: x[1]['orthogonality_score'])
                    print(f"  最佳正交性: {best_ortho[0].upper()} (分数: {best_ortho[1]['orthogonality_score']:.6f})")

                    best_robust = max(self.experiment_results['method_comparison'].items(),
                                   key=lambda x: x[1]['robustness_score'])
                    print(f"  最佳鲁棒性: {best_robust[0].upper()} (分数: {best_robust[1]['robustness_score']:.6f})")

        return self.experiment_results


def main():
    """主函数"""
    print("开始异构张量空间数值验证实验...")

    # 创建实验套件
    experiment_suite = TensorSpaceExperimentSuite(seed=42)

    # 运行完整实验
    results = experiment_suite.run_full_experiment_suite()

    print(f"\n实验完成!结果已保存到'tensor_space_experiments_results.png'")
    print("=" * 80)


if __name__ == "__main__":
    main()