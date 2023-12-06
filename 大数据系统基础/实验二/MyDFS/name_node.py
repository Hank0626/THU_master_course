import math
import os
import socket

import numpy as np
import pandas as pd

from common import *


# NameNode功能
# 1. 保存文件的块存放位置信息
# 2. ls ： 获取文件/目录信息
# 3. get_fat_item： 获取文件的FAT表项
# 4. new_fat_item： 根据文件大小创建FAT表项
# 5. rm_fat_item： 删除一个FAT表项
# 6. format: 删除所有FAT表项

class NameNode:
    def run(self):  # 启动NameNode
        # 创建一个监听的socket
        listen_fd = socket.socket()
        try:
            # 监听端口
            listen_fd.bind(("0.0.0.0", NAME_NODE_PORT))
            listen_fd.listen(5)
            print("Name node start")
            while True:
                # 等待连接，连接后返回通信用的套接字
                sock_fd, addr = listen_fd.accept()
                print("connected by {}".format(addr))
                
                try:
                    # 获取请求方发送的指令
                    request = str(sock_fd.recv(128), encoding='utf-8')
                    request = request.split()  # 指令之间使用空白符分割
                    print("Request: {}".format(request))
                    
                    cmd = request[0]  # 指令第一个为指令类型
                    
                    if cmd == "ls":  # 若指令类型为ls, 则返回DFS上对于文件、文件夹的内容
                        dfs_path = request[1]  # 指令第二个参数为DFS目标地址
                        response = self.ls(dfs_path)
                    elif cmd == "get_fat_item":  # 指令类型为获取FAT表项
                        dfs_path = request[1]  # 指令第二个参数为DFS目标地址
                        response = self.get_fat_item(dfs_path)
                    elif cmd == "new_fat_item":  # 指令类型为新建FAT表项
                        dfs_path = request[1]  # 指令第二个参数为DFS目标地址
                        file_size = int(request[2])
                        response = self.new_fat_item(dfs_path, file_size)
                    elif cmd == "rm_fat_item":  # 指令类型为删除FAT表项
                        dfs_path = request[1]  # 指令第二个参数为DFS目标地址
                        response = self.rm_fat_item(dfs_path)
                    elif cmd == "format":
                        response = self.format()
                    else:  # 其他位置指令
                        response = "Undefined command: " + " ".join(request)
                    
                    print("Response: {}".format(response))
                    sock_fd.send(bytes(response, encoding='utf-8'))
                except KeyboardInterrupt:  # 如果运行时按Ctrl+C则退出程序
                    break
                except Exception as e:  # 如果出错则打印错误信息
                    print(e)
                finally:
                    sock_fd.close()  # 释放连接
        except KeyboardInterrupt:  # 如果运行时按Ctrl+C则退出程序
            pass
        except Exception as e:  # 如果出错则打印错误信息
            print(e)
        finally:
            listen_fd.close()  # 释放连接
    
    def ls(self, dfs_path):
        local_path = os.path.join(NAME_NODE_DIR, dfs_path)
        # 如果不存在，返回错误信息
        if not os.path.exists(local_path):
            return "No such file or directory: {}".format(dfs_path)
        
        if os.path.isdir(local_path):
            # 如果目标地址是一个文件夹，则显示该文件夹下内容
            dirs = os.listdir(local_path)
            response = " ".join(dirs)
        else:
            # 如果目标是文件则显示文件的FAT表信息
            with open(local_path) as f:
                response = f.read()
        
        return response
    
    def get_fat_item(self, dfs_path):
        # 获取FAT表内容
        local_path = os.path.join(NAME_NODE_DIR, dfs_path)
        response = pd.read_csv(local_path)
        return response.to_csv(index=False)
    
    def new_fat_item(self, dfs_path, file_size):
        nb_blks = int(math.ceil(file_size / DFS_BLK_SIZE))
        print(file_size, nb_blks)
        
        # todo 如果dfs_replication为复数时可以新增host_name的数目
        data_pd = pd.DataFrame(columns=['blk_no', 'host_name', 'blk_size'])

        counter = 0
        for i in range(nb_blks):
            blk_size = min(DFS_BLK_SIZE, file_size - i * DFS_BLK_SIZE)
            # 循环选择存放位置
            if DFS_REPLICATION == 1:
                host_name = HOST_LIST[counter]
                counter = 0 if counter + 1 == len(HOST_LIST) else counter + 1
                data_pd.loc[i] = [i, host_name, blk_size]
            
            # 副本数大于1时，随机选择存放位置
            else:
                host_name_list = np.random.choice(HOST_LIST, size=DFS_REPLICATION, replace=False)
                for j in range(DFS_REPLICATION): 
                    host_name = host_name_list[j]
                    data_pd.loc[(i - 1) * DFS_REPLICATION + j] = [i, host_name, blk_size]
        
        # 获取本地路径
        local_path = os.path.join(NAME_NODE_DIR, dfs_path)
        # 若目录不存在则创建新目录
        os.system("mkdir -p {}".format(os.path.dirname(local_path)))
        # 保存FAT表为CSV文件
        data_pd.to_csv(local_path, index=False)
        # 同时返回CSV内容到请求节点
        return data_pd.to_csv(index=False)
    
    def rm_fat_item(self, dfs_path):
        local_path = NAME_NODE_DIR + dfs_path
        response = pd.read_csv(local_path)
        os.remove(local_path)
        return response.to_csv(index=False)
    
    def format(self):
        format_command = "rm -rf {}/*".format(NAME_NODE_DIR)
        os.system(format_command)
        return "Format namenode successfully~"

        
    
# 创建NameNode并启动
name_node = NameNode()
name_node.run()


