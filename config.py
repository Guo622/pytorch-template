import json
import torch


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