import os
import socket
import pickle
import base64
import numpy as np
import threading

from common import *


# DataNode支持的指令有:
# 1. load 加载数据块
# 2. store 保存数据块
# 3. rm 删除数据块
# 4. format 删除所有数据块

class DataNode:
    def run(self):
        # 创建一个监听的socket
        listen_fd = socket.socket()
        try:
            # 监听端口
            listen_fd.bind(("0.0.0.0", DATA_NODE_PORT))
            listen_fd.listen(5)
            print("Data node start")
            while True:
                # 等待连接，连接后返回通信用的套接字
                sock_fd, addr = listen_fd.accept()
                print("Received request from {}".format(addr))
                
                try:
                    # 获取请求方发送的指令
                    request = str(sock_fd.recv(BUF_SIZE), encoding='utf-8')
                     # 如果指令是heartbeat

                    request = request.split()  # 指令之间使用空白符分割
                    print(request)
                        
                    cmd = request[0]  # 指令第一个为指令类型
                        
                    if cmd == "load":  # 加载数据块
                        dfs_path = request[1]  # 指令第二个参数为DFS目标地址
                        response = self.load(dfs_path)
                    elif cmd == "store":  # 存储数据块
                        dfs_path = request[1]  # 指令第二个参数为DFS目标地址
                        response = self.store(sock_fd, dfs_path)
                    elif cmd == "rm":  # 删除数据块
                        dfs_path = request[1]  # 指令第二个参数为DFS目标地址
                        response = self.rm(dfs_path)
                    elif cmd == "format":  # 格式化DFS
                        response = self.format()
                    elif cmd == "info":
                        input_path = request[1]
                        response = self.matrixInfo(input_path)
                    elif cmd == "map":
                        input_path = request[1]
                        start = int(request[2])
                        end = int(request[3])
                        response = self.mapper(input_path, start, end)
                    else:
                        response = "Undefined command: " + " ".join(request)

                    sock_fd.send(bytes(response, encoding='utf-8'))
                except KeyboardInterrupt:
                    break
                finally:
                    sock_fd.close()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(e)
        finally:
            listen_fd.close()
    
    def load(self, dfs_path):
        # 本地路径
        local_path = os.path.join(DATA_NODE_DIR, dfs_path)
        # 读取本地数据
        with open(local_path) as f:
            chunk_data = f.read(DFS_BLK_SIZE)
        
        return chunk_data
    
    def store(self, sock_fd, dfs_path):
        # 从Client获取块数据
        chunk_data = sock_fd.recv(BUF_SIZE)
        # 本地路径
        local_path = os.path.join(DATA_NODE_DIR, dfs_path)
        # 若目录不存在则创建新目录
        os.system("mkdir -p {}".format(os.path.dirname(local_path)))
        # 将数据块写入本地文件
        with open(local_path, "wb") as f:
            f.write(chunk_data)
        
        return "Store chunk {} successfully~".format(local_path)
    
    def rm(self, dfs_path):
        local_path = os.path.join(DATA_NODE_DIR, dfs_path)
        rm_command = "rm -rf " + local_path
        os.system(rm_command)
        
        return "Remove chunk {} successfully~".format(local_path)
    
    def format(self):
        format_command = "rm -rf {}/*".format(DATA_NODE_DIR)
        os.system(format_command)
        
        return "Format datanode successfully~"
    
    def matrixInfo(self, input_path):
        local_path = os.path.join(DATA_NODE_DIR, input_path)
        with open(local_path, 'r') as file:
            lines = file.readlines()
    
        m, n, p = tuple(map(int, lines[0].strip().split(' ')))

        matrix_len = len(lines) - 1
        
        return f"{m} {n} {p} {matrix_len}"

    def process_line_segment(self, lines, m, p, res, lock):
        for line in lines:
            r, c, v = int(line[1]), int(line[2]), int(line[3])
            # 对于A来说，每一个r，要出现p次，每一次是 (r, i) = (c, v)
            # 对于B来说，每一个c，要出现m次，每一次是 (i, c) = (r, v)
            if line[0] == 'A':
                for i in range(1, p + 1):
                    with lock:
                        res[(r, i)] = res.get((r, i), []) + [(0, c, v)]
            else:
                for i in range(1, m + 1):
                    with lock:
                        res[(i, c)] = res.get((i, c), []) + [(1, r, v)]

    def mapper(self, input_path, start, end):
        local_path = os.path.join(DATA_NODE_DIR, input_path)

        lines = []
        with open(local_path) as f:
            for current_line_number, line in enumerate(f):
                # 获取m, n, p信息
                if current_line_number == 0:
                    m, n, p = tuple(map(int, line.strip().split(' ')))
                if current_line_number > start:
                    lines.append(line.strip().split(','))
                if current_line_number == end:
                    break

        # 并行进行mapper操作
        num_threads = os.cpu_count() - 1
        num_threads = min(len(lines), num_threads)
        
        print(num_threads)
        
        lines_per_thread = len(lines) // num_threads
        res = {}
        lock = threading.Lock()

        threads = []
        for i in range(num_threads):
            start = i * lines_per_thread
            end = (i + 1) * lines_per_thread if i != num_threads - 1 else len(lines)
            thread = threading.Thread(target=self.process_line_segment, args=(lines[start: end], m, p, res, lock))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 用base64和pickle来将map的字典转化为字符串形式
        return base64.b64encode(pickle.dumps(res)).decode('utf-8')


# 创建DataNode对象并启动
data_node = DataNode()
data_node.run()
