import torch
from heterogeneous_tensor_space import HeterogeneousTensorSystem

# 测试参数
test_params = {'n': 64, 'K': 4, 'V': 32, 'c': 256}

# 创建CCA系统
system = HeterogeneousTensorSystem(**test_params, method='cca')

# 拟合模型
print("拟合CCA模型...")
A_train = torch.randn(100, test_params['n'])
B_train = torch.randn(100, test_params['K'], test_params['V']).view(100, -1)
system.transformer.fit(A_train, B_train)

# 生成测试数据
A = torch.randn(1, test_params['n'])
B = torch.randn(1, test_params['K'], test_params['V'])

try:
    # 前向变换
    C = system.forward(A, B)
    print(f"前向变换成功: A{A.shape} + B{B.shape} -> C{C.shape}")

    # 检查C的内容
    print(f"C的维度: {C.shape}")

    # 分解测试
    n_components = min(test_params['n'], test_params['K'] * test_params['V'], test_params['c'] // 2)
    print(f"n_components: {n_components}")
    print(f"d_A: {test_params['n']}, d_B: {test_params['K'] * test_params['V']}")

    print(f"expansion_A shape: {system.transformer.expansion_A.shape}")
    print(f"expansion_B shape: {system.transformer.expansion_B.shape}")

    C_A = C[:, :n_components]
    C_B = C[:, n_components:2*n_components]
    print(f"C_A shape: {C_A.shape}")
    print(f"C_B shape: {C_B.shape}")

    # 手动测试逆变换步骤
    C_A = C[:, :64]
    C_B = C[:, 64:128]

    print(f"C_A shape: {C_A.shape}")
    print(f"C_B shape: {C_B.shape}")
    print(f"expansion_A shape: {system.transformer.expansion_A.shape}")
    print(f"expansion_B shape: {system.transformer.expansion_B.shape}")

    # A的重构
    A_test = C_A @ torch.pinverse(system.transformer.expansion_A.T)
    print(f"A_test shape: {A_test.shape}")

    # B的重构
    pinv_exp_B = torch.pinverse(system.transformer.expansion_B)
    print(f"pinv_exp_B shape: {pinv_exp_B.shape}")

    B_flat_test = C_B @ pinv_exp_B[:, :64].T  # 只使用前64列
    print(f"B_flat_test shape: {B_flat_test.shape}")

    # 逆变换
    A_recon, B_recon = system.inverse(C)
    print(f"逆变换成功: -> A{A_recon.shape}, B{B_recon.shape}")

except Exception as e:
    print(f"错误: {str(e)}")
    import traceback
    traceback.print_exc()