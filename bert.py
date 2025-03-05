import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json
import re
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def judge_emotion(input_file):
    # 读取CSV文件
    df_train = pd.read_csv('virus_train1.csv')

    # 假设CSV文件有两列：'comment'表示评论内容，'label'表示评论的情绪标签
    # 你可以根据实际情况调整这部分代码

    comments = df_train['comment'].tolist()
    labels = df_train['label'].tolist()

    # 标签编码：将情感标签转换为数字
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)  # 将情感标签转为数字

    # 初始化BERT模型和tokenizer
    model_name = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=7)  # 7种情感类别


    # 定义数据集类
    class SentimentDataset(Dataset):
        def __init__(self, comments, labels, tokenizer, max_len=128):
            self.comments = comments
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.comments)

        def __getitem__(self, idx):
            comment = self.comments[idx]
            label = self.labels[idx]
            encoding = self.tokenizer.encode_plus(
                comment,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(label, dtype=torch.long)
            }


    # 划分训练集和验证集
    train_comments, val_comments, train_labels, val_labels = train_test_split(comments, encoded_labels, test_size=0.1)

    # 创建数据加载器
    train_dataset = SentimentDataset(train_comments, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_comments, val_labels, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=3e-5)

    epochs = 3  # 可以根据需要调整
    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} complete. Loss: {loss.item()}")

    # 保存模型
    model.save_pretrained("sentiment_model")
    tokenizer.save_pretrained("sentiment_model")


    # 使用训练好的模型对新的评论进行预测
    def predict_sentiment(comments):
        model.eval()
        predictions = []
        with torch.no_grad():
            for comment in comments:
                encoding = tokenizer.encode_plus(
                    comment,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)  # 使用softmax得到每种情感的概率
                predictions.append(probs.squeeze().cpu().numpy())

        return predictions



    # 加载 emoji 字典
    with open('data/emoji_Chinese.json', 'r', encoding='utf-8') as f:
        emoji_list = json.load(f)

    # 将列表转换为字典形式
    emoji_dict = {list(item.keys())[0]: list(item.values())[0] for item in emoji_list}


    # 替换评论中的 emoji 为对应的文字
    def replace_emoji_with_text(text):
        # 使用正则表达式找到评论中的 emoji
        for emoji, meaning in emoji_dict.items():
            # 替换 emoji 为对应的文字
            text = text.replace(emoji, meaning)
        return text
    # 计算积极与消极情绪的比例
    def calculate_sentiment_ratio(emotion_dict):
        positive_emotions = ['like', 'surprise', 'happy']  # 假设这三个是积极情绪
        negative_emotions = ['angry', 'disgust', 'fear', 'sad']  # 这些是消极情绪

        positive_score = sum(emotion_dict[emotion] for emotion in positive_emotions)
        negative_score = sum(emotion_dict[emotion] for emotion in negative_emotions)

        total_score = positive_score + negative_score
        if total_score > 0:
            positive_ratio = positive_score / total_score
            negative_ratio = negative_score / total_score
        else:
            positive_ratio = 0
            negative_ratio = 0

        return positive_ratio, negative_ratio
    # 测试用数据
    # 读取测试CSV文件
    df_test = pd.read_csv(input_file)  # 假设测试数据在一个名为'test_comments.csv'的文件中
    df_test['content'] = df_test['content'].apply(replace_emoji_with_text)
    test_comments = df_test['content'].tolist()  # 读取评论内容
    # test_comments = ['今天天气不错','个真的很糟糕']
    review = df_test.copy()
    # 预测
    probs = predict_sentiment(test_comments)

    results = []
    pos_neg=[]
    for idx, prob in enumerate(probs):
        emotion_dict = {label: p for label, p in zip(label_encoder.classes_, prob)}
        total_prob = prob.sum()  # 确保总和为1

        # 计算情感的积极和消极比例
        positive_ratio, negative_ratio = calculate_sentiment_ratio(emotion_dict)
        pos_neg_dict = {'positive':positive_ratio,'negative':negative_ratio}
        # 保存结果
        result = emotion_dict.copy()
        results.append(result)
        pos_neg.append(pos_neg_dict)

    review['sentiment']=results
    review['pos-neg'] = pos_neg
    # 将结果保存为CSV
    results_df = pd.DataFrame(review)
    results_df.to_csv('emotion_prediction_wh.csv', index=False,encoding='utf-8-sig')
    print("情感分析结果已保存到 'emotion_predictions.csv'")


    def calculate_metrics(predictions, true_labels):
        # 将预测的概率最大值索引作为预测标签
        predicted_labels = np.argmax(predictions, axis=1)

        # 计算整体 Accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)

        # 计算每个类别的 Precision, Recall, F1
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=None)

        # 计算加权平均的 Precision, Recall, F1
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

        # 组织返回数据
        overall_metrics = {
            "accuracy": accuracy,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted
        }

        per_class_metrics = {
            "precision": precision_per_class,
            "recall": recall_per_class,
            "f1": f1_per_class
        }

        return overall_metrics, per_class_metrics


    def evaluate_model(model, val_dataloader, device):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                    # 获取模型的输出
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)  # 得到每种情感的概率

                    # 将预测的概率转换为标签
                preds = probs.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

            # 计算各种指标
        overall_metrics, per_class_metrics = calculate_metrics(np.array(all_preds), np.array(all_labels))

        return overall_metrics, per_class_metrics

    # 在训练完后进行验证集的评估
    overall_metrics, per_class_metrics = evaluate_model(model, val_dataloader, device)

    # 输出整体评价指标
    print(f"Validation Accuracy: {overall_metrics['accuracy'] * 100:.2f}%")
    print(f"Validation Precision (Weighted): {overall_metrics['precision_weighted'] * 100:.2f}%")
    print(f"Validation Recall (Weighted): {overall_metrics['recall_weighted'] * 100:.2f}%")
    print(f"Validation F1 Score (Weighted): {overall_metrics['f1_weighted'] * 100:.2f}%")

    # 输出每种情绪类别的 Precision、Recall、F1
    print("\nPer-Class Metrics:")
    for i, (p, r, f) in enumerate(zip(per_class_metrics['precision'], per_class_metrics['recall'], per_class_metrics['f1'])):
        print(f"Class {i} - Precision: {p:.4f}, Recall: {r:.4f}, F1-score: {f:.4f}")
