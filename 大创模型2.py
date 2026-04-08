import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import joblib


# 文本数据预处理
def preprocess_text_data(file_path):
    data = pd.read_csv(file_path)
    # 1. 检查重复行
    duplicate_rows = data[data.duplicated(subset='id')]
    if not duplicate_rows.empty:
        print("存在重复行：")
        print(duplicate_rows)
        # 删除重复行
        data = data.drop_duplicates(subset='id')
    else:
        print("不存在重复行。")

    # 2. 处理缺失值
    # 检查各列缺失值情况
    missing_values = data.isnull().sum()
    print("各列缺失值数量：")
    print(missing_values)

    # 对于location列中的'unknown'，可以考虑用合适的值填充，这里用'unknown_fill'替代
    data['location'] = data['location'].replace('unknown', 'unknown_fill')

    # 处理text列的缺失值，将NaN替换为空字符串
    data['text'] = data['text'].fillna('')

    # 3. 日期处理
    # 假设文本中包含日期信息，例如 'january 13'
    # 这里只是示例，需要根据实际情况调整
    def extract_date(text):
        import re
        pattern = r'([a-zA-Z]+)\s+(\d+)'
        match = re.search(pattern, text)
        if match:
            month = match.group(1)
            day = match.group(2)
            # 可以将月份转换为数字
            month_dict = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12
            }
            if month.lower() in month_dict:
                month_num = month_dict[month.lower()]
                return f'2025-{month_num:02d}-{int(day):02d}'
        return None

    data['date'] = data['text'].apply(extract_date)

    # 4. 文本长度标准化
    scaler = MinMaxScaler()
    data['text_length_scaled'] = scaler.fit_transform(data[['text_length']])

    # 5. 选择特征和目标变量
    # 假设除了'target'列和一些不需要的列（如'id', 'text'等），其他列作为特征
    columns_to_drop = ['id', 'text', 'date']  # 可以根据实际情况添加不需要的列

    # 找出包含字符串的列
    string_columns = data.select_dtypes(include=['object']).columns
    columns_to_drop.extend(string_columns)

    X_text = data.drop(columns=columns_to_drop + ['target'])
    y_text = data['target']

    return X_text, y_text


# 图片数据预处理
def preprocess_image_data(file_path):
    df = pd.read_csv(file_path)
    # 重新提取分类标签，假设灾难类型在路径分隔符后的第 6 个元素（索引为 5）
    df['label'] = df['image_path'].apply(lambda x: x.split('\\')[5])

    # 准备特征和标签
    X_image = df.drop(['image_path', 'label'], axis=1)
    y_image = df['label']

    return X_image, y_image


# 多模态融合
def multimodal_fusion(text_file_path, image_file_path):
    # 预处理文本和图片数据
    X_text, y_text = preprocess_text_data(text_file_path)
    X_image, y_image = preprocess_image_data(image_file_path)

    # 确保文本数据和图片数据样本数量一致
    min_samples = min(len(X_text), len(X_image))
    X_text = X_text.iloc[:min_samples]
    y_text = y_text.iloc[:min_samples]
    X_image = X_image.iloc[:min_samples]

    # 合并特征
    X = pd.concat([X_text.reset_index(drop=True), X_image.reset_index(drop=True)], axis=1)
    y = y_text

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 训练逻辑回归模型
    model_logreg = LogisticRegression()
    model_logreg.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred_logreg = model_logreg.predict(X_test)

    # 计算准确率
    accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
    print(f"多模态融合模型（逻辑回归）准确率: {accuracy_logreg:.2f}")
    # 保存逻辑回归模型到指定位置
    save_path = r"C:\Users\26093\Desktop\大创项目\解题"
    joblib.dump(model_logreg, f'{save_path}/multimodal_logreg_model.joblib')

    # 创建随机森林分类器
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"多模态融合模型（随机森林）准确率: {accuracy_rf:.2f}")
    # 保存随机森林模型到指定位置
    joblib.dump(model_rf, f'{save_path}/multimodal_rf_model.joblib')

    # 创建 SVM 分类器
    model_svm = SVC(random_state=42)
    model_svm.fit(X_train, y_train)
    y_pred_svm = model_svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f"多模态融合模型（SVM）准确率: {accuracy_svm:.2f}")
    # 保存 SVM 模型到指定位置
    joblib.dump(model_svm, f'{save_path}/multimodal_svm_model.joblib')

    # 创建多层感知器分类器
    model_mlp = MLPClassifier(random_state=42, max_iter=500)
    model_mlp.fit(X_train, y_train)
    y_pred_mlp = model_mlp.predict(X_test)
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    print(f"多模态融合模型（多层感知器）准确率: {accuracy_mlp:.2f}")
    # 保存多层感知器模型到指定位置
    joblib.dump(model_mlp, f'{save_path}/multimodal_mlp_model.joblib')


# 调用函数进行多模态融合
text_file_path = r'C:\Users\26093\Desktop\python\tweets_features.csv'
image_file_path = r'C:\Users\26093\Desktop\python\disaster_image_features.csv'
multimodal_fusion(text_file_path, image_file_path)
    