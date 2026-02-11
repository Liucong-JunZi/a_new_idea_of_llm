import torch
from heterogeneous_tensor_space import HeterogeneousTensorSystem

# 测试参数
test_params = {'n': 32, 'K': 4, 'V': 16, 'c': 128}

# 创建对比学习系统
system = HeterogeneousTensorSystem(**test_params, method='contrastive')

# 生成测试数据
A = torch.randn(1, test_params['n'])
B = torch.randn(1, test_params['K'], test_params['V'])

try:
    # 前向变换
    C = system.forward(A, B)
    print(f"前向变换成功: A{A.shape} + B{B.shape} -> C{C.shape}")
    print(f"C的内容维度: {C.shape}")

    # 逆变换
    A_recon, B_recon = system.inverse(C)
    print(f"逆变换成功: C{C.shape} -> A{A_recon.shape}, B{B_recon.shape}")

except Exception as e:
    print(f"错误: {str(e)}")
    import traceback
    traceback.print_exc()