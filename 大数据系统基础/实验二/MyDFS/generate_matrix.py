import numpy as np
import sys


def generate(m, n, p, max_val, input_path, output_path):
    total_elements = m * n + n * p      # 计算总元素数
    max_non_zero = total_elements // 1000 * 6  # 规定非零元素的最大数量

    # 初始化矩阵
    A = np.zeros((m, n), dtype=int)
    B = np.zeros((n, p), dtype=int)

    # 随机设置非零元素
    non_zero_count = 0
    while non_zero_count < max_non_zero:
        i, j = np.random.randint(0, m), np.random.randint(0, n)
        if A[i, j] == 0:
            A[i, j] = np.random.randint(1, max_val)
            non_zero_count += 1
        
        i, j = np.random.randint(0, n), np.random.randint(0, p)
        if B[i, j] == 0:
            B[i, j] = np.random.randint(1, max_val)
            non_zero_count += 1

    # 写入输入文件
    with open(input_path, 'w') as fi:
        fi.write(f"{m} {n} {p}\n")
        for i in range(m):
            for j in range(n):
                if A[i, j] != 0:
                    fi.write(f"A,{i+1},{j+1},{A[i, j]}\n")
        
        for i in range(n):
            for j in range(p):
                if B[i, j] != 0:
                    fi.write(f"B,{i+1},{j+1},{B[i, j]}\n")

    # 计算结果并写入输出文件
    res = A @ B
    with open(output_path, 'w') as fo:
        for i in range(m):
            for j in range(p):
                if res[i, j] != 0:
                    fo.write(f"{i+1},{j+1},{res[i, j]}\n")
    
    print("Generation Done!!")

argv = sys.argv
argc = len(argv) - 1

if argc == 6:
    try:
        m = int(argv[1])
        n = int(argv[2])
        p = int(argv[3])
        max_val = int(argv[4])
        input_path = argv[5]
        output_path = argv[6]
        generate(m, n, p, max_val, input_path, output_path)
    except TypeError:
        print("Usage: python generate_matrix.py <A_rows> <A_cols> <B_cols> <max_val> <input_path> <output_path>")
else:
    print("Usage: python generate_matrix.py <A_rows> <A_cols> <B_cols> <max_val> <input_path> <output_path>")
