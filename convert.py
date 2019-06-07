import json
import math
import time
import glob
import os
import datetime

#####################################################

# 输入文件(星号表示匹配任意字符串)
INPUT_FILE = './*_keypoints.json'

# 输出文件
OUTPUT_FILE = './output.txt'

# 点的索引
NODE_IDS = [[4, 5],
            [13, 14],
            [16, 17],
            [19, 20]]

# 读取间隔(以秒为单位)
TIME_INTERVAL = 1


#####################################################


def compute_angle(vec1, vec2):
    """
    compute the angle between two vectors

    vec1: List[float]
    vec2: List[float]

    returns: float in [0, 2*pi]
    """
    dot, l1, l2 = 0, 0, 0
    for x1, x2 in zip(vec1, vec2):
        dot += x1 * x2
        l1 += x1**2
        l2 += x2**2
    cos = dot / math.sqrt(l1 * l2)
    angle = math.acos(cos)
    angle = angle * (180 / math.pi)
    return angle


def convert(infile_list, outfile, node_ids):
    outputs = []
    for infile in infile_list:
        # 将json文件读取为python字典
        with open(infile, 'r') as fin:
            dic = json.load(fin)

        # 点坐标
        nums = dic['people'][0]['pose_keypoints_2d']
        points = [[nums[x - 1], nums[y - 1]] for x, y in node_ids]

        # 边向量
        edges = [[points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1]] for i in range(len(points) - 1)]
        # 加入x轴
        edges = [[1, 0]] + edges

        # 向量夹角
        angles = [compute_angle(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

        # 格式化为2位小数的字符串
        formatted = ','.join(['{:.2f}'.format(x) for x in angles])
        outputs.append(formatted)

    # 输出到txt文件
    with open(outfile, 'a') as fout:
        for line in outputs:
            fout.write(line + '\n')


def file2id(filename):
    # 从完整路径名中提取文件名
    filename = os.path.basename(filename)

    # 返回文件编号
    return int(filename.split('_')[0])


def main():
    # 最后处理的文件编号
    last_processed = -1

    # 新建输出文件
    with open(OUTPUT_FILE, 'w') as fout:
        pass

    while True:
        # 匹配所有json文件
        files = glob.glob(INPUT_FILE)

        # 将文件按序号排序
        files = sorted(files, key=file2id)

        # 去除已经处理过的文件
        files = [f for f in files if file2id(f) > last_processed]
        if len(files) > 0:
            last_processed = file2id(files[-1])
            convert(files, OUTPUT_FILE, NODE_IDS)
            print('{} - processed {}'.format(datetime.datetime.now(), files))

        time.sleep(TIME_INTERVAL)

if __name__ == '__main__':
    main()
