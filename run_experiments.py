#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验运行入口脚本
Entry Point for Running Experiments
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from heterogeneous_tensor_space import HeterogeneousTensorSystem
from tensor_space_experiments import TensorSpaceExperimentSuite

def main():
    """主函数"""
    print("=" * 80)
    print("异构张量空间正交嵌入系统")
    print("Heterogeneous Tensor Space Orthogonal Embedding System")
    print("=" * 80)

    # 检查CUDA可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")

    # 基础测试
    print("\n1. 基础功能验证")
    print("-" * 40)

    test_params = {'n': 64, 'K': 4, 'V': 32, 'c': 256}
    print(f"测试参数: {test_params}")

    for method in ['parametric', 'cca', 'contrastive']:
        print(f"\n测试 {method.upper()} 方法:")

        # 创建系统
        system = HeterogeneousTensorSystem(**test_params, method=method)

        # 生成测试数据
        A = torch.randn(1, test_params['n'])
        B = torch.randn(1, test_params['K'], test_params['V'])

        try:
            # CCA方法需要先拟合
            if method == 'cca':
                print("  [INFO] 拟合CCA模型...")
                # 生成训练数据
                n_train = 100
                A_train = torch.randn(n_train, test_params['n'])
                B_train = torch.randn(n_train, test_params['K'], test_params['V']).view(n_train, -1)
                system.transformer.fit(A_train, B_train)
                print("  [OK] CCA模型拟合完成")

            # 前向变换
            C = system.forward(A, B)
            print(f"  [OK] 前向变换: A{A.shape} + B{B.shape} -> C{C.shape}")

            # 逆变换
            A_recon, B_recon = system.inverse(C)
            print(f"  [OK] 逆变换: C{C.shape} -> A{A_recon.shape}, B{B_recon.shape}")

            # 计算误差
            recon_error = torch.nn.functional.mse_loss(A_recon, A) + torch.nn.functional.mse_loss(B_recon, B)
            print(f"  [OK] 重构误差: {recon_error.item():.8f}")

        except Exception as e:
            print(f"  [FAIL] 测试失败: {str(e)}")

    # 运行完整实验套件
    print("\n\n2. 运行完整实验套件")
    print("-" * 40)

    try:
        experiment_suite = TensorSpaceExperimentSuite(seed=42)
        results = experiment_suite.run_full_experiment_suite()

        print("\n实验成功完成!")

        # 保存结果摘要
        save_experiment_summary(results)

    except Exception as e:
        print(f"\n实验套件运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

def save_experiment_summary(results):
    """保存实验摘要"""
    print("\n3. 保存实验摘要")
    print("-" * 40)

    try:
        summary_text = "异构张量空间实验摘要\n"
        summary_text += "=" * 50 + "\n\n"

        if 'method_comparison' in results:
            summary_text += "方法性能比较:\n"
            summary_text += "-" * 30 + "\n"

            methods = list(results['method_comparison'].keys())
            for method in methods:
                metrics = results['method_comparison'][method]
                summary_text += f"\n{method.upper()}方法:\n"
                summary_text += f"  重构误差: {metrics['reconstruction_error']:.8f}\n"
                summary_text += f"  正交性分数: {metrics['orthogonality_score']:.6f}\n"
                summary_text += f"  鲁棒性分数: {metrics['robustness_score']:.6f}\n"
                summary_text += f"  计算时间: {metrics['computation_time']:.3f}秒\n"

        # 保存到文件
        with open('experiment_summary.txt', 'w', encoding='utf-8') as f:
            f.write(summary_text)

        print("[OK] 实验摘要已保存到 'experiment_summary.txt'")

    except Exception as e:
        print(f"[FAIL] 保存摘要失败: {str(e)}")

def demo_quick_test():
    """快速演示"""
    print("\n\n3. 快速演示")
    print("-" * 40)

    # 创建一个简单的异构张量系统
    n, K, V, c = 32, 4, 16, 128
    system = HeterogeneousTensorSystem(n, K, V, c, method='parametric')

    # 生成示例数据
    A = torch.randn(1, n)
    B = torch.randn(1, K, V)

    print(f"输入:")
    print(f"  语义向量 A: 形状{A.shape}, 范数[{torch.norm(A).item():.4f}]")
    print(f"  声学张量 B: 形状{B.shape}, 范数[{torch.norm(B).item():.4f}]")

    # 执行变换
    C = system.forward(A, B)
    print(f"\n统一空间 C: 形状{C.shape}, 范数[{torch.norm(C).item():.4f}]")

    # 逆变换
    A_recon, B_recon = system.inverse(C)
    print(f"\n重构结果:")
    print(f"  A_recon: 形状{A_recon.shape}, 误差[{torch.nn.functional.mse_loss(A_recon, A).item():.8f}]")
    print(f"  B_recon: 形状{B_recon.shape}, 误差[{torch.nn.functional.mse_loss(B_recon, B).item():.8f}]")

    # 验证正交性
    if hasattr(system.transformer, 'semantic_encoder'):
        B_flat = B.view(1, K * V)
        sem_feat = system.transformer.semantic_encoder(A)
        ac_feat = system.transformer.acoustic_encoder(B_flat)
        sem_proj = system.transformer.semantic_proj(sem_feat)
        ac_proj = system.transformer.acoustic_proj(ac_feat)
        ortho_score = system.transformer.check_orthogonality(sem_proj, ac_proj)
        print(f"\n正交性分数: {ortho_score:.6f} (越小越好)")

    print("\n[OK] 快速演示完成!")

if __name__ == "__main__":
    main()
    demo_quick_test()

    print("\n" + "=" * 80)
    print("所有测试和实验完成!")
    print("=" * 80)