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


if __name__ == "__main__":
    generate_annotation()
