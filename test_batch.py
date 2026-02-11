import torch
from heterogeneous_tensor_space import HeterogeneousTensorSystem

# 测试参数
test_params = {'n': 32, 'K': 4, 'V': 16, 'c': 128}

# 测试对比学习批量处理
system = HeterogeneousTensorSystem(**test_params, method='contrastive')

# 生成批量测试数据
batch_size = 8
A_batch = torch.randn(batch_size, test_params['n'])
B_batch = torch.randn(batch_size, test_params['K'], test_params['V'])

try:
    # 批量前向变换
    C_batch = system.forward(A_batch, B_batch)
    print(f"批量前向变换成功: A{A_batch.shape} + B{B_batch.shape} -> C{C_batch.shape}")

    # 批量逆变换
    A_recon, B_recon = system.inverse(C_batch)
    print(f"批量逆变换成功: C{C_batch.shape} -> A{A_recon.shape}, B{B_recon.shape}")

    # 检查维度
    assert A_recon.shape == A_batch.shape
    assert B_recon.shape == B_batch.shape
    print("维度匹配正确!")

    # 计算重构误差
    recon_error = torch.nn.functional.mse_loss(A_recon, A_batch) + torch.nn.functional.mse_loss(B_recon, B_batch)
    print(f"批量重构误差: {recon_error.item():.6f}")

except Exception as e:
    print(f"错误: {str(e)}")
    import traceback
    traceback.print_exc()