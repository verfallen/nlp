from glob import glob
import os
import random
import pandas as pd
from config import *

# 根据标注文件生成对应关系
def get_annotation(ann_path):
    with open(ann_path) as file:
        anns = {}

        for line in file.readlines():
            arr = line.split("\t")[1].split()
            name = arr[0]
            start = int(arr[1])
            end = int(arr[-1])

            # 过长的标注，可能有问题
            if end - start > 50:
                continue
            anns[start] = "B-" + name
            for i in range(start + 1, end):
                anns[i] = "I-" + name

        return anns


def get_text(text_path):
    with open(text_path) as file:
        return file.read()


def generate_annotation():
    for text_path in glob(ORIGIN_DIR + "*.txt"):
        print(text_path)
        text = get_text(text_path)
        anns = get_annotation(text_path[:-3] + "ann")

        df = pd.DataFrame({"word": list(text), "label": "O"})
        df.loc[anns.keys(), "label"] = list(anns.values())

        filename = os.path.split(text_path)[-1]
        df.to_csv(ANNOTATION_DIR + filename, header=None, index=None)


# 拆分训练集，测试集
def split_sample(test_size=0.3):
    files = glob(ANNOTATION_DIR + "*.txt")
    random.seed(0)
    random.shuffle(files)

    test_num = int(len(files) * test_size)
    test_files = files[:test_num]
    train_files = files[test_num:]

    merge_files(test_files, TEST_SAMPLE_PATH)
    merge_files(train_files, TRAIN_SAMPLE_PATH)


# 将多个文件合并到一个指定文件中
def merge_files(files, target_path):
    with open(target_path, "a") as f:
        for file in files:
            print(file)
            text = open(file).read()
            f.write(text)


if __name__ == "__main__":
    # 生成一对一标注
    generate_annotation()

    # 拆分并生成数据集
    split_sample()
