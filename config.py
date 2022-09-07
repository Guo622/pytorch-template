import json
import torch
import argparse

class Config():
    def __init__(self) -> None:

        self.seed = 42
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_data_path = './data/snli_1.0/snli_1.0_train.txt'
        self.val_data_path = './data/snli_1.0/snli_1.0_dev.txt'
        self.test_data_path = './data/snli_1.0/snli_1.0_test.txt'

        self.vocab_path = './vocab/vocab.txt'
        self.log_path = './log/'
        self.save_path = './save/'

        self.min_freq = 10    # 5->16310  10->12406
        self.sentence1_max_len = 20  # max 84 min 4 mean 16  >20 108756 >25 41295
        self.sentence2_max_len = 12  # max 65 min 3 mean 10  >12 106968 >15 36587
        self.vocab_size = 12406
        self.pad_token_id = 0
        self.num_classes = 3

        self.weight_decay = 0.01
        self.learning_rate = 1e-3 
        self.adam_epsilon = 1e-6
        self.warmup_ratio = 0.05 
        self.train_batch_size = 256
        self.val_batch_size = 1024
        self.test_batch_size = 1024
        self.max_epochs = 15
        self.print_steps = 20

        self.max_position_embeddings = 25
        self.embedding_size = 768
        self.hidden_size = 384
        self.dropout = 0.5
        self.layer_norm_eps = 1e-5
        

    def save_dict(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=4)

    def load_from_dict(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            dic = json.load(f)
        for key, value in dic.items():
            self.__setattr__(key,value)


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')

    # ========================= Data Configs ==========================
    parser.add_argument('--train_annotation', type=str, default='/opt/ml/input/data/annotations/labeled.json')
    parser.add_argument('--pretrain_annotation', type=str, default='/opt/ml/input/data/annotations/unlabeled.json')
    parser.add_argument('--test_annotation', type=str, default='/opt/ml/input/data/annotations/test.json')
    parser.add_argument('--train_zip_frames', type=str, default='/opt/ml/input/data/zip_frames/labeled/')
    parser.add_argument('--pretrain_zip_frames', type=str, default='/opt/ml/input/data/zip_frames/unlabeled/')
    parser.add_argument('--test_zip_frames', type=str, default='/opt/ml/input/data/zip_frames/test/')
    parser.add_argument('--test_output_csv', type=str, default='/opt/ml/output/result.csv')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=32, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=32, type=int, help="use for validation duration per worker")
    parser.add_argument('--pretrain_batch_size', default=32, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=60, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=8, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='save/')
    parser.add_argument('--ckpt_file', type=str, default='save/model.bin')
    parser.add_argument('--best_score', default=0.5, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=5, help='How many epochs')
    parser.add_argument('--max_steps', default=20000, type=int, metavar='N', help='number of total epochs to run')  #50000
    parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.") #0.01
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    # ========================== Swin ===================================
    parser.add_argument('--swin_pretrained_path', type=str, default='opensource_models/swin_tiny_patch4_window7_224.pth')

    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='opensource_models/chinese-macbert-base')
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    parser.add_argument('--bert_seq_length', type=int, default=84)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)

    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--bert_embedding_size', type=int, default=768)
    parser.add_argument('--lstm_hidden_size', type=int, default=1024)
    parser.add_argument('--max_frames', type=int, default=12)
    parser.add_argument('--vlad_cluster_size', type=int, default=64)
    parser.add_argument('--vlad_groups', type=int, default=4)
    parser.add_argument('--vlad_hidden_size', type=int, default=1024, help='nextvlad output size using dense')
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')

    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=512, help="linear size before final linear")

    return parser.parse_args()