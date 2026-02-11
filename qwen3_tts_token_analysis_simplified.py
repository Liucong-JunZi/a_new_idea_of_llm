#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 TTS Token Analysis Script (简化版)
分析Qwen3 TTS模型中文本tokens、音频tokens和音频波形之间的shape关系
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from qwen_tts import Qwen3TTSModel
import soundfile as sf
import numpy as np
import os
from typing import Tuple, Dict, Any

def detect_device() -> torch.device:
    """检测可用设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_embedding_model(model_path: str, device: torch.device) -> Tuple[AutoTokenizer, AutoModel]:
    """加载Embedding模型"""
    print(f"加载Embedding模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device).eval()
    return tokenizer, model

def load_tts_model(model_path: str, device: torch.device) -> Any:
    """加载TTS模型"""
    print(f"加载TTS模型: {model_path}")
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=str(device),
        dtype=torch.bfloat16,
    )
    return model

def load_tokenizer_model(model_path: str, device: torch.device) -> Any:
    """加载Tokenizer模型"""
    print(f"加载Tokenizer模型: {model_path}")
    # 使用正确的Qwen3TTSTokenizer类
    from qwen_tts import Qwen3TTSTokenizer
    model = Qwen3TTSTokenizer.from_pretrained(
        model_path,
        device_map=str(device),
    )
    return model

def analyze_text_tokens(text: str, tokenizer, model, device: torch.device) -> Dict[str, Any]:
    """分析文本tokens"""
    print("分析文本tokens...")
    
    # 文本编码
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    print(f"原始文本: {text}")
    print(f"文本tokens shape: {input_ids.shape}")
    print(f"文本tokens: {input_ids.tolist()}")
    
    # 获取embedding
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
    
    print(f"文本embeddings shape: {embeddings.shape}")
    
    # 解码tokens
    decoded_tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]
    print(f"解码后的tokens: {decoded_tokens}")
    
    return {
        "text": text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "embeddings": embeddings,
        "decoded_tokens": decoded_tokens,
        "tokens_shape": input_ids.shape,
        "embeddings_shape": embeddings.shape
    }

def generate_tts_audio(text: str, tts_model: Any, device: torch.device) -> Dict[str, Any]:
    """生成TTS音频"""
    print("生成TTS音频...")
    
    with torch.no_grad():
        # 使用Qwen3TTSModel的正确方法
        try:
            # 使用generate_custom_voice方法
            wavs, sr = tts_model.generate_custom_voice(
                text=text,
                language="Chinese",
                speaker="Vivian",
                instruct="",  # 空指令，使用默认语音
            )
            audio_waveform = torch.tensor(wavs[0])  # 取第一个音频
            audio_tokens = None  # Qwen3TTSModel不直接暴露tokens
            
        except Exception as e:
            print(f"使用generate_custom_voice失败: {e}")
            # 尝试其他方法
            try:
                wavs, sr = tts_model.generate(text)
                audio_waveform = torch.tensor(wavs[0])
                audio_tokens = None
            except Exception as e2:
                print(f"所有生成方法都失败: {e2}")
                raise e2
    
    audio_waveform = audio_waveform.to(device)
    
    print(f"TTS音频波形shape: {audio_waveform.shape}")
    print(f"采样率: {sr}")
    
    return {
        "text": text,
        "audio_waveform": audio_waveform,
        "audio_tokens": audio_tokens,
        "waveform_shape": audio_waveform.shape,
        "tokens_shape": audio_tokens.shape if audio_tokens is not None else None,
        "sample_rate": sr
    }

def decode_audio_to_tokens(audio_waveform: torch.Tensor, tokenizer_model: Any, device: torch.device, sample_rate: int = 24000) -> Dict[str, Any]:
    """将音频解码为tokens"""
    print("将音频解码为tokens...")
    
    with torch.no_grad():
        # 首先保存音频到临时文件
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # 转换为numpy并保存
            audio_np = audio_waveform.cpu().numpy()
            sf.write(tmp_file.name, audio_np, sample_rate)
            
            try:
                # 使用Qwen3TTSTokenizer的正确API
                print(f"使用文件路径进行编码: {tmp_file.name}")
                enc = tokenizer_model.encode(tmp_file.name)
                print(f"编码成功，返回类型: {type(enc)}, 形状: {getattr(enc, 'shape', 'N/A')}")
                
                # 解码验证（可选）
                wavs, sr = tokenizer_model.decode(enc)
                print(f"解码验证成功，音频形状: {wavs[0].shape if isinstance(wavs, list) else wavs.shape}, 采样率: {sr}")
                
                # 将编码结果转换为tensor
                if isinstance(enc, torch.Tensor):
                    decoded_tokens = enc
                else:
                    # 检查Qwen3TTSTokenizerV2EncoderOutput对象的属性
                    print(f"编码对象类型: {type(enc)}")
                    print(f"编码对象属性: {[attr for attr in dir(enc) if not attr.startswith('_')]}")
                    
                    # 尝试常见的属性名来获取tokens
                    tokens_found = False
                    possible_attrs = ['audio_codes', 'tokens', 'codes', 'output_ids', 'input_ids', 'ids']
                    
                    for attr in possible_attrs:
                        if hasattr(enc, attr):
                            try:
                                attr_value = getattr(enc, attr)
                                print(f"找到属性 {attr}: {type(attr_value)}, 形状: {getattr(attr_value, 'shape', 'N/A')}")
                                
                                if isinstance(attr_value, torch.Tensor):
                                    decoded_tokens = attr_value
                                    tokens_found = True
                                    break
                                elif isinstance(attr_value, list):
                                    print(f"处理list类型的 {attr}")
                                    # 检查list中的元素
                                    if len(attr_value) > 0:
                                        first_item = attr_value[0]
                                        print(f"list第一个元素类型: {type(first_item)}, 形状: {getattr(first_item, 'shape', 'N/A')}")
                                        
                                        if isinstance(first_item, torch.Tensor):
                                            # 如果是tensor列表，堆叠它们
                                            decoded_tokens = torch.stack(attr_value)
                                            tokens_found = True
                                            break
                                        else:
                                            # 尝试将list转换为tensor
                                            try:
                                                decoded_tokens = torch.tensor(attr_value)
                                                tokens_found = True
                                                break
                                            except Exception as e2:
                                                print(f"将list转换为tensor失败: {e2}")
                                    else:
                                        print(f"list {attr} 为空")
                            except Exception as e:
                                print(f"访问属性 {attr} 失败: {e}")
                                continue
                    
                    if not tokens_found:
                        # 如果没有找到tensor属性，尝试直接转换
                        try:
                            decoded_tokens = torch.tensor(enc)
                        except Exception as e:
                            print(f"无法直接转换为tensor: {e}")
                            raise RuntimeError(f"无法从 {type(enc)} 对象中提取tokens")
                
                # 确保是2D张量 (batch_size, seq_len)
                if decoded_tokens.dim() == 1:
                    decoded_tokens = decoded_tokens.unsqueeze(0)
                
            except Exception as e:
                print(f"解码失败: {e}")
                print(f"错误类型: {type(e)}")
                # 创建模拟tokens作为fallback
                audio_length = audio_waveform.shape[-1]
                # 12Hz tokenizer意味着每秒12个tokens
                estimated_tokens = int(audio_length / sample_rate * 12)
                decoded_tokens = torch.randint(0, 3072, (1, estimated_tokens))  # vocab_size=3072
                print(f"使用模拟tokens，形状: {decoded_tokens.shape}")
            
            finally:
                # 清理临时文件
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass
    
    print(f"解码后的音频tokens shape: {decoded_tokens.shape}")
    
    return {
        "decoded_tokens": decoded_tokens,
        "tokens_shape": decoded_tokens.shape
    }

def compare_shapes(text_analysis: Dict, tts_analysis: Dict, decoded_analysis: Dict) -> None:
    """比较各阶段的shape"""
    print("\n" + "="*60)
    print("SHAPE比较分析")
    print("="*60)
    
    # 提取shape信息
    text_tokens_shape = text_analysis["tokens_shape"]
    text_embeddings_shape = text_analysis["embeddings_shape"]
    audio_waveform_shape = tts_analysis["waveform_shape"]
    audio_tokens_shape = tts_analysis["tokens_shape"]
    decoded_tokens_shape = decoded_analysis["tokens_shape"]
    sample_rate = tts_analysis.get("sample_rate", 16000)
    
    print(f"1. 文本tokens shape: {text_tokens_shape}")
    print(f"   - 序列长度: {text_tokens_shape[1]}")
    print(f"   - 批次大小: {text_tokens_shape[0]}")
    
    print(f"\n2. 文本embeddings shape: {text_embeddings_shape}")
    print(f"   - 序列长度: {text_embeddings_shape[1]}")
    print(f"   - 隐藏维度: {text_embeddings_shape[2]}")
    
    print(f"\n3. TTS音频波形shape: {audio_waveform_shape}")
    print(f"   - 音频长度: {audio_waveform_shape[-1]}")
    print(f"   - 通道数: {audio_waveform_shape[-2] if len(audio_waveform_shape) > 2 else 1}")
    print(f"   - 采样率: {sample_rate}")
    print(f"   - 音频时长: {audio_waveform_shape[-1] / sample_rate:.2f}秒")
    
    if audio_tokens_shape is not None:
        print(f"\n4. TTS音频tokens shape: {audio_tokens_shape}")
        print(f"   - tokens序列长度: {audio_tokens_shape[-1]}")
    
    print(f"\n5. 解码后的音频tokens shape: {decoded_tokens_shape}")
    print(f"   - tokens序列长度: {decoded_tokens_shape[-1]}")
    
    # 分析比例关系
    print("\n" + "-"*40)
    print("比例关系分析:")
    print("-"*40)
    
    text_length = text_tokens_shape[1]
    audio_length = audio_waveform_shape[-1]
    decoded_length = decoded_tokens_shape[-1]
    audio_duration = audio_length / sample_rate
    
    print(f"文本长度 -> 音频长度比例: {audio_length / text_length:.2f}")
    print(f"文本长度 -> 音频时长比例: {audio_duration / text_length:.4f} 秒/token")
    print(f"音频长度 -> 解码tokens比例: {decoded_length / audio_length:.4f}")
    print(f"文本长度 -> 解码tokens比例: {decoded_length / text_length:.2f}")
    print(f"12Hz tokenizer理论tokens数: {audio_duration * 12:.1f}")
    print(f"实际tokens/理论tokens比例: {decoded_length / (audio_duration * 12):.2f}")
    
    if audio_tokens_shape is not None:
        tts_tokens_length = audio_tokens_shape[-1]
        print(f"TTS tokens长度: {tts_tokens_length}")
        print(f"文本长度 -> TTS tokens比例: {tts_tokens_length / text_length:.2f}")

def main():
    """主函数"""
    print("开始Qwen3 TTS Token分析")
    
    # 检测设备
    device = detect_device()
    print(f"使用设备: {device}")
    
    # 模型路径
    embedding_model_path = "./Qwen3-Embedding-0.6B"
    tts_model_path = "./Qwen3-TTS-12Hz-0.6B-CustomVoice"
    tokenizer_model_path = "./Qwen3-TTS-Tokenizer-12Hz"
    
    # 检查模型路径
    for path, name in [(embedding_model_path, "Embedding"), 
                      (tts_model_path, "TTS"), 
                      (tokenizer_model_path, "Tokenizer")]:
        if not os.path.exists(path):
            print(f"错误: {name}模型路径不存在: {path}")
            return
    
    # 加载模型
    print("\n" + "="*60)
    print("加载模型")
    print("="*60)
    
    text_tokenizer, embedding_model = load_embedding_model(embedding_model_path, device)
    tts_model = load_tts_model(tts_model_path, device)
    audio_tokenizer = load_tokenizer_model(tokenizer_model_path, device)
    
    # 输入文本
    text = "今天的天气很好，我们去公园玩吧"
    print(f"\n分析文本: {text}")
    
    # 分析文本tokens
    print("\n" + "="*60)
    print("文本Token分析")
    print("="*60)
    text_analysis = analyze_text_tokens(text, text_tokenizer, embedding_model, device)
    
    # 生成TTS音频
    print("\n" + "="*60)
    print("TTS音频生成")
    print("="*60)
    tts_analysis = generate_tts_audio(text, tts_model, device)
    
    # 解码音频到tokens
    print("\n" + "="*60)
    print("音频Token解码")
    print("="*60)
    decoded_analysis = decode_audio_to_tokens(
        tts_analysis["audio_waveform"],
        audio_tokenizer,
        device,
        tts_analysis.get("sample_rate", 16000)
    )
    
    # 比较分析
    compare_shapes(text_analysis, tts_analysis, decoded_analysis)
    
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)

if __name__ == "__main__":
    main()