ORIGIN_DIR = "./input/origin/ruijin_round1_train2_20181022/"
ANNOTATION_DIR = "./output/annotation/"
TRAIN_SAMPLE_PATH = "./output/train_sample.txt"
TEST_SAMPLE_PATH = "./output/test_sample.txt"

VOCAB_PATH = "./output/vocab.txt"
LABEL_PATH = "./output/label.txt"

WORD_PAD = "<PAD>"
WORD_UNK = "<UNK>"
WORD_PAD_ID = 1
WORD_UNK_ID = 0
LABEL_O_ID = 0

VOCAB_SIZE = 3000

# 模型配置
EMBEDDING_DIM = 100
HIDDEN_SIZE = 256
TARGET_SIZE = 31
LR = 1e-3
EPOCH = 5

MODEL_DIR = "./output/model/"
