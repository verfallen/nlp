import re

def rm_sign(string):
    string = re.sub("[\.\!_,\$\(\)\"\'\]\[！!\?，。？、~@#￥……&]+", "", string) 
    return string


def load_data(file_path = './data/corpus.txt'):
    with open(file_path,'r')as f:
        for line in f:
            line = line.strip()
            if(len(line) == 0):
                continue
            yield rm_sign(line.lower()).split()
                
