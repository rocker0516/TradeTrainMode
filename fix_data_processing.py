#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

def fix_train_transformer():
    """修正train_transformer.py中的數據處理邏輯"""
    
    # 讀取文件
    with open('train_transformer.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 要替換的舊內容
    old_section = '''    # 加載數據
    print("加載數據...")
    df = pd.read_csv(args.data_path)
    print(f"數據量: {len(df)} 行")
    
    # 創建環境
    print("創建環境...")'''
    
    # 新的內容
    new_section = '''    # 加載數據
    print("加載數據...")
    df = pd.read_csv(args.data_path)
    print(f"數據量: {len(df)} 行")
    
    # 數據預處理
    print("預處理數據...")
    if 'timestamp' in df.columns:
        # 將時間戳設為索引，這樣env的時間特徵處理函數可以正常工作
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        print("✓ 時間戳已設為索引")
    
    # 確保所有列都是數值類型
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"警告: 列 '{col}' 不是數值類型，將嘗試轉換")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"✓ 最終數據形狀: {df.shape}")
    print(f"✓ 數據列: {list(df.columns)}")
    
    # 創建環境
    print("創建環境...")'''
    
    # 替換內容
    if old_section in content:
        content = content.replace(old_section, new_section)
        print("✓ 找到並替換了目標代碼段")
    else:
        print("❌ 未找到目標代碼段，嘗試手動定位...")
        # 顯示相關行以便調試
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '加載數據' in line:
                print(f"第{i+1}行: {line}")
                for j in range(max(0, i-2), min(len(lines), i+8)):
                    print(f"  {j+1}: {lines[j]}")
                break
        return False
    
    # 寫回文件
    with open('train_transformer.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ train_transformer.py 已更新")
    return True

if __name__ == "__main__":
    fix_train_transformer() 