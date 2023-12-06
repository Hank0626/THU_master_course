DFS_REPLICATION = 1  # 每个数据块会存在多少个机器上
DFS_BLK_SIZE = 4096  # 数据块大小

# NameNode和DataNode数据存放位置
NAME_NODE_DIR = "./dfs/name/"
DATA_NODE_DIR = "./dfs/data/"

NAME_NODE_PORT = 21009  # NameNode监听端口
DATA_NODE_PORT = 11009  # DataNode程序监听端口

# 集群中的主机列表
# HOST_LIST = ['localhost']
# HOST_LIST = ['thumm01', 'thumm02', 'thumm03', 'thumm04']
HOST_LIST = ['thumm01']
NAME_NODE_HOST = 'localhost'

BUF_SIZE = DFS_BLK_SIZE * 2
