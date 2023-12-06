import os
import socket
import time
from io import StringIO

import threading
from queue import Queue

import pickle
import base64

import pandas as pd
import numpy as np

from common import *


class Client:
    def __init__(self):
        self.name_node_sock = socket.socket()
        self.name_node_sock.connect((NAME_NODE_HOST, NAME_NODE_PORT))
    
    def __del__(self):
        self.name_node_sock.close()
    
    def ls(self, dfs_path):
        # TODO: 向NameNode发送请求，查看dfs_path下文件或者文件夹信息
        try:
            request = "ls {}".format(dfs_path)
            self.name_node_sock.send(bytes(request, encoding='utf-8'))
            print(str(self.name_node_sock.recv(BUF_SIZE), encoding='utf-8'))
        except Exception as e:
            print(e)
        finally:
            pass

    def copyFromLocal(self, local_path, dfs_path):
        file_size = os.path.getsize(local_path)
        print("File size: {}".format(file_size))
        
        request = "new_fat_item {} {}".format(dfs_path, file_size)
        print("Request: {}".format(request))
        
        # 从NameNode获取一张FAT表
        self.name_node_sock.send(bytes(request, encoding='utf-8'))
        fat_pd = self.name_node_sock.recv(BUF_SIZE)

        # 打印FAT表，并使用pandas读取
        fat_pd = str(fat_pd, encoding='utf-8')
        print("Fat: \n{}".format(fat_pd))
        fat = pd.read_csv(StringIO(fat_pd))
        
        # 根据FAT表逐个向目标DataNode发送数据块
        fp = open(local_path)
        blk_no = -1
        for idx, row in fat.iterrows():
            if row['blk_no'] != blk_no:
                blk_no = row['blk_no']
                data = fp.read(int(row['blk_size']))

            data_node_sock = socket.socket()
            data_node_sock.connect((row['host_name'], DATA_NODE_PORT))
            blk_path = dfs_path + ".blk{}".format(row['blk_no'])
            
            request = "store {}".format(blk_path)
            data_node_sock.send(bytes(request, encoding='utf-8'))
            time.sleep(0.1)  # 两次传输需要间隔一段时间，避免粘包
            data_node_sock.send(bytes(data, encoding='utf-8'))
            data_node_sock.close()
        fp.close()
    
    def copyToLocal(self, dfs_path, local_path):
        request = "get_fat_item {}".format(dfs_path)
        print("Request: {}".format(request))
        # TODO: 从NameNode获取一张FAT表；打印FAT表；根据FAT表逐个从目标DataNode请求数据块，写入到本地文件中

        self.name_node_sock.send(bytes(request, encoding='utf-8'))
        fat_pd = self.name_node_sock.recv(BUF_SIZE)
        
        # 打印FAT表，并使用pandas读取
        fat_pd = str(fat_pd, encoding='utf-8')
        print("Fat: \n{}".format(fat_pd))
        fat = pd.read_csv(StringIO(fat_pd))
        
        # 根据FAT表逐个从目标DataNode请求数据块，写入到本地文件中
        fp = open(local_path, 'w')
        
        blk_no = -1
        for idx, row in fat.iterrows():
            if row['blk_no'] != blk_no:
                blk_no = row['blk_no']
            else:
                continue
            
            data_node_sock = socket.socket()
            data_node_sock.connect((row['host_name'], DATA_NODE_PORT))
            blk_path = dfs_path + ".blk{}".format(row['blk_no'])
            
            request = "load {}".format(blk_path)
            data_node_sock.send(bytes(request, encoding='utf-8'))
            time.sleep(0.2)  # 两次传输需要间隔一段时间，避免粘包
            data = str(data_node_sock.recv(BUF_SIZE), encoding='utf-8')
            fp.write(data)
            data_node_sock.close()
        fp.close()

    def rm(self, dfs_path):
        request = "rm_fat_item {}".format(dfs_path)
        print("Request: {}".format(request))
        # TODO: 从NameNode获取改文件的FAT表，获取后删除；打印FAT表；根据FAT表逐个告诉目标DataNode删除对应数据块

        self.name_node_sock.send(bytes(request, encoding='utf-8'))
        fat_pd = self.name_node_sock.recv(BUF_SIZE)
        
        # 打印FAT表；根据FAT表逐个告诉目标DataNode删除对应数据块
        fat_pd = str(fat_pd, encoding='utf-8')
        print("Fat: \n{}".format(fat_pd))
        fat = pd.read_csv(StringIO(fat_pd))
        
        for idx, row in fat.iterrows():
            data_node_sock = socket.socket()
            data_node_sock.connect((row['host_name'], DATA_NODE_PORT))
            blk_path = dfs_path + ".blk{}".format(row['blk_no'])
            
            request = "rm {}".format(blk_path)
            data_node_sock.send(bytes(request, encoding='utf-8'))
            print(str(data_node_sock.recv(BUF_SIZE), encoding='utf-8'))
            data_node_sock.close()

    def format(self):
        request = "format"
        print(request)
        
        self.name_node_sock.send(bytes(request, encoding='utf-8'))
        print(str(self.name_node_sock.recv(BUF_SIZE), encoding='utf-8'))
        
        for host in HOST_LIST:
            data_node_sock = socket.socket()
            data_node_sock.connect((host, DATA_NODE_PORT))
            
            data_node_sock.send(bytes("format", encoding='utf-8'))
            print(str(data_node_sock.recv(BUF_SIZE), encoding='utf-8'))
            
            data_node_sock.close()

    # 并行分布式调用data_node的mapper操作
    def worker(self, host, start, end, input_path, res_map):
        print(f"Calling host {host} to map...")
        data_node_sock = socket.socket()
        data_node_sock.connect((host, DATA_NODE_PORT))
        request = "map {} {} {}".format(input_path, start, end)
        data_node_sock.send(bytes(request, encoding='utf-8'))

        # 防止包传不完整
        data = b''
        while True:
            packet = data_node_sock.recv(BUF_SIZE)
            if not packet:
                break
            data += packet
        data_node_sock.close()

        # 用pickle来load base64解码之后的map结果
        host_map = pickle.loads(base64.b64decode(data))
        
        # shuffle操作
        with threading.Lock():
            for k, v in host_map.items():
                res_map[k] = res_map.get(k, []) + v

    # 进行分布式MapReduce的矩阵计算
    def matrixCal(self, input_path, output_path):
        # 选取第一个host来获取矩阵大小
        data_node_sock = socket.socket()
        data_node_sock.connect((HOST_LIST[0], DATA_NODE_PORT))
        # 在data_node里读取文件第一行，获取 m,n,p,matrix_len
        data_node_sock.send(bytes(f"info {input_path}", encoding='utf-8'))
        response_msg = str(data_node_sock.recv(BUF_SIZE), encoding='utf-8')
        m, n, p, matrix_len = tuple(map(int, response_msg.split(' ')))

        num_host = len(HOST_LIST)
        res_map = {}
        

        start_time = time.time()
        # 如果矩阵行数太少了，直接单机
        if num_host > matrix_len:   
            data_node_sock = socket.socket()
            data_node_sock.connect((HOST_LIST[0], DATA_NODE_PORT))
            request = "map {} {} {}".format(input_path, 0, matrix_len)
            data_node_sock.send(bytes(request, encoding='utf-8'))
            data = b''
            while True:
                packet = data_node_sock.recv(BUF_SIZE)
                if not packet:
                    break
                data += packet
            data_node_sock.close()
            res_map = pickle.loads(base64.b64decode(data))
        else:
            # 否则用threading并行调用多机器
            host_len = matrix_len // num_host
            threads = []
            for idx, host in enumerate(HOST_LIST):
                start = idx * host_len
                end = (idx + 1) * host_len if (idx + 1) * host_len < matrix_len else matrix_len
                thread = threading.Thread(target=self.worker, args=(host, start, end, input_path, res_map))
                threads.append(thread)
                thread.start()

            for t in threads:
                t.join()

        print(f"Time Cost: {time.time() - start_time:.2f}s")
        
        # 每个thread单独计算一部分shuffle过后的集合
        def process(subset, result_queue):
            partial_results = []
            
            # 对每一个key计算取值，用pandas加速
            for key in sorted(subset):
                array = subset[key]
                df = pd.DataFrame(array, columns=['matrix', 'key', 'value'])
                if len(df['matrix'].unique()) == 1:
                    continue
                
                grouped = df.groupby(['matrix', 'key'])
                multiplication = grouped['value'].prod().unstack(fill_value=0)
                sum_result = (multiplication.iloc[0] * multiplication.iloc[1]).sum()
                
                if sum_result != 0:
                    partial_results.append([key[0], key[1], sum_result])

            result_queue.put(partial_results)

        # 用多个thread进行reducer
        def reducer(data):
            data_keys = sorted(data.keys())
            num_threads = min(len(data_keys), os.cpu_count() - 1)
            keys_per_thread = len(data_keys) // num_threads
            # queue保证所有thread的结果有序被保存
            result_queue = Queue()

            threads = []
            for i in range(num_threads):
                start = i * keys_per_thread
                end = start + keys_per_thread if i != num_threads - 1 else len(data_keys)
                subset = {key: data[key] for key in data_keys[start:end]}
                thread = threading.Thread(target=process, args=(subset, result_queue))
                threads.append(thread)
                thread.start()

            # 等待所有线程完成
            for thread in threads:
                thread.join()

            # 从队列中收集结果并合并
            all_results = []
            while not result_queue.empty():
                all_results.extend(result_queue.get())
                
            # 保证输出按照矩阵index的顺序
            all_results = sorted(all_results, key=lambda x: (x[0], x[1]))

            return np.array(all_results)

        # 保存最终的结果
        np.savetxt(output_path, reducer(res_map), fmt="%d", delimiter=',')


# 解析命令行参数并执行对于的命令
import sys

argv = sys.argv
argc = len(argv) - 1

client = Client()

cmd = argv[1]
if cmd == '-ls':
    if argc == 2:
        dfs_path = argv[2]
        client.ls(dfs_path)
    else:
        print("Usage: python client.py -ls <dfs_path>")
elif cmd == "-rm":
    if argc == 2:
        dfs_path = argv[2]
        client.rm(dfs_path)
    else:
        print("Usage: python client.py -rm <dfs_path>")
elif cmd == "-copyFromLocal":
    if argc == 3:
        local_path = argv[2]
        dfs_path = argv[3]
        client.copyFromLocal(local_path, dfs_path)
    else:
        print("Usage: python client.py -copyFromLocal <local_path> <dfs_path>")
elif cmd == "-copyToLocal":
    if argc == 3:
        dfs_path = argv[2]
        local_path = argv[3]
        client.copyToLocal(dfs_path, local_path)
    else:
        print("Usage: python client.py -copyFromLocal <dfs_path> <local_path>")
elif cmd == "-format":
    client.format()
elif cmd == "-matrixCal":
    if argc == 3:
        input_path = argv[2]
        output_path = argv[3]
        client.matrixCal(input_path, output_path)
    else:
        print("Usage: python client.py -matrixCal <input_path> <output_path>")
else:
    print("Undefined command: {}".format(cmd))
    print("Usage: python client.py <-ls | -copyFromLocal | -copyToLocal | -rm | -format> other_arguments")
