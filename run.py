from textcnn import TextCNN
from torchtext.vocab import Vectors
from torchtext.data import BucketIterator
from torch import nn, optim
import torch
import random
import jieba_fast
from tqdm import tqdm
import os
import numpy as np
import json
import time
import onnxruntime

from dataset import get_iflytek_dataset

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
root = os.path.dirname(os.path.abspath(__file__))
print(root)


class ModelCls:
    def __init__(
        self, dim=300, device="cpu", batch_size=1024, lr=5e-3, epochs=5, max_length=400
    ):
        self.dim = dim
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.max_length = max_length

    def train(self, pretrained_file):
        sentence, label, train, dev = get_iflytek_dataset(self.max_length)
        sentence.vocab.load_vectors(
            vectors=Vectors(os.path.join(root, pretrained_file))
        )
        model = TextCNN(
            sentence.vocab.vectors.shape[0],  # 词个数
            sentence.vocab.vectors.shape[1],  # 向量维度
            len(label.vocab.stoi),  # 标签个数
        )
        model.embeddings.weight.data.copy_(sentence.vocab.vectors)
        datas = (train, dev)
        train_iter, val_iter = BucketIterator.splits(
            datas,
            device=self.device,
            repeat=False,
            batch_size=self.batch_size,
            sort=False,
            shuffle=True,
        )
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        epochs = []
        train_accs = []
        val_accs = []
        train_epoch_loss_list = []
        val_epoch_loss_list = []
        for epoch in tqdm(range(1, self.epochs + 1)):
            epochs.append(epoch)
            model.train()

            train_epoch_loss = 0
            y_p = []
            y_t = []
            for batch, batch_data in enumerate(train_iter):
                x = batch_data.sentence
                y = batch_data.label
                optimizer.zero_grad()
                proba = model.forward(x[0])
                loss = criterion(proba, y)
                loss.backward()
                optimizer.step()
                y_p += proba.max(1)[1]
                y_t += list(y.data)
                train_epoch_loss += float(loss.data)
            correct = sum([1 if i == j else 0 for i, j in zip(y_p, y_t)])
            accuracy = correct / len(y_p)
            train_accs.append(round(accuracy, 3))
            train_epoch_loss_list.append(round(train_epoch_loss / batch + 1, 3))
            # 验证集测试
            val_epoch_loss = 0
            model.eval()
            y_p = []
            y_t = []
            for batch, batch_data in enumerate(val_iter):
                x = batch_data.sentence
                y = batch_data.label
                proba = model.forward(x[0])
                y_p += proba.max(1)[1]
                y_t += list(y.data)
                val_epoch_loss += float(loss.data)
            correct = sum([1 if i == j else 0 for i, j in zip(y_p, y_t)])
            accuracy = correct / len(y_p)
            val_accs.append(round(accuracy, 3))
            val_epoch_loss_list.append(round(val_epoch_loss / batch + 1, 3))
        with open("metric.md", "w") as f:
            f.write("## 这是一个CML自动跑的textcnn模型，训练和评测iflytek公开数据集\n\n")
            f.write("---\n")
            f.write("+ 训练过程\n\n")
            f.write("|epoch|训练集loss|验证集loss|训练集acc|验证集acc|\n")
            f.write("|-|-|-|-|-|\n")
            for epoch, train_loss, val_loss, train_acc, val_acc in zip(
                epochs, train_epoch_loss_list, val_epoch_loss_list, train_accs, val_accs
            ):
                f.write(f"|{epoch}|{train_loss}|{val_loss}|{train_acc}|{val_acc}|\n")

        # 存onnx格式
        dummy = torch.zeros(self.max_length, 1).long()
        torch.onnx.export(
            model.to("cpu"),
            dummy,
            f="model.onnx",
            export_params=True,
            verbose=False,
            opset_version=12,
            training=False,
            do_constant_folding=False,
            input_names=["input"],
            output_names=["output"],
        )
        # 存字典
        with open("vocab.json", "w") as f:
            json.dump(dict(sentence.vocab.stoi), f, indent=4, ensure_ascii=False)
        # 存标签
        with open("reverse_label.json", "w") as f:
            label_dict = label.vocab.stoi
            reverse_label_dict = dict(zip(label_dict.values(), label_dict.keys()))
            json.dump(reverse_label_dict, f, indent=4, ensure_ascii=False)

    def load(self):
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = 1
        self.model = onnxruntime.InferenceSession("model.onnx", sess_options)
        with open("vocab.json", "r") as f:
            self.vocab = json.load(f)
        with open("reverse_label.json", "r") as f:
            self.reverse_label = json.load(f)

    def predict(self, sentence):
        # 编码
        inputs = []
        for word in jieba_fast.lcut(sentence):
            if word in self.vocab.keys():
                inputs.append([self.vocab[word]])
            else:
                inputs.append([self.vocab["<unk>"]])

        # 强行padding至最大长度
        if len(inputs) <= self.max_length:
            inputs = inputs + [[self.vocab["<pad>"]]] * (self.max_length - len(inputs))
        else:
            inputs = inputs[: self.max_length]
        # 输入转换格式
        onnx_inputs = {"input": np.array(inputs)}
        # onnx推断
        logits = self.model.run(None, onnx_inputs)[0]
        return self.reverse_label[str(logits.argmax(1)[0])]

    def evaluate(self):
        # 批测流程
        self.load()
        with open("dataset/dev.json", "r") as f:
            start = time.time()
            gold_labels = []
            pred_labels = []
            for line in tqdm(f.readlines()):
                record = json.loads(line)
                gold_labels.append(record["label"])
                pred_labels.append(self.predict(record["sentence"]))
            end = time.time()
            waste_time = end - start
            waste_every_record = round(waste_time * 1000 / len(gold_labels), 3)
            qps = round(len(gold_labels) / waste_time * 1000, 3)
            correct = sum(
                [1 if i == j else 0 for i, j in zip(gold_labels, pred_labels)]
            )
            accuracy = round(correct / len(pred_labels), 3)
        with open("evalate.md", "w") as f:
            f.write("\n---\n")
            f.write("+ 预测\n\n")
            f.write("|指标|值|\n")
            f.write("|-|-|\n")
            f.write(f"|单条用时（ms）|{waste_every_record}|\n")
            f.write(f"|单秒执行条数|{qps}|\n")
            f.write(f"|准确率|{accuracy}|\n")
            f.write("---\n")
            f.write("+ 参考\n\t+ [cml官网](https://cml.dev/)\n")
            f.write("---\n")

        return round(accuracy, 3)


if __name__ == "__main__":
    m = ModelCls(device="cpu", max_length=400, epochs=3)
    m.train("sgns_vec_split4iflytek")
    print(m.evaluate())
