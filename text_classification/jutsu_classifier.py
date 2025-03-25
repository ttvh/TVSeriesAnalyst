import pandas as pd
import torch
import huggingface_hub
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          DataCollatorWithPadding,
                          TrainingArguments,
                          pipeline)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datasets import Dataset
import gc
from .cleaner import Cleaner
from .training_utils import get_class_weights, compute_metrics
from .custom_trainer import CustomTrainer
import os
import sys
import pathlib

folder_path = pathlib.Path().parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))

class JutsuClassifier():
    def __init__(self,
                 model_path,                # Đường dẫn tới mô hình
                 data_path=None,            # Đường dẫn tới dữ liệu, mặc định là None
                 text_column_name='text',   # Tên cột văn bản
                 label_column_name='jutsu', # Tên cột nhãn
                 model_name="distilbert/distilbert-base-uncased", # Tên mô hình mặc định
                 test_size=0.2,             # Kích thước tập kiểm tra
                 num_labels=3,              # Số lượng nhãn
                 huggingface_token=None):   # Token Hugging Face, nếu có

        self.model_path = model_path
        self.data_path = data_path
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.model_name = model_name
        self.test_size = test_size
        self.num_labels = num_labels
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # Thiết bị: CUDA nếu có, nếu không thì CPU

        self.huggingface_token = huggingface_token
        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token) # Đăng nhập Hugging Face nếu có token

        self.tokenizer = self.load_tokenizer() # Tải tokenizer

        if not self.model_path or not huggingface_hub.repo_exists(self.model_path):
            if data_path is None:
                raise ValueError(
                    "Đường dẫn dữ liệu là bắt buộc để huấn luyện mô hình, vì đường dẫn mô hình không tồn tại trên Hugging Face Hub")
            
            train_data, test_data = self.load_data(self.data_path) # Tải dữ liệu huấn luyện và kiểm tra
            train_data_df = train_data.to_pandas()
            test_data_df = test_data.to_pandas()

            all_data = pd.concat([train_data_df, test_data_df]).reset_index(drop=True)
            class_weights = get_class_weights(all_data) # Tính trọng số lớp

            self.train_model(train_data, test_data, class_weights) # Huấn luyện mô hình

        self.model = self.load_model(self.model_path) # Tải mô hình đã huấn luyện

    def load_model(self, model_path):
        if not model_path:
            model_path = self.model_name  # Sử dụng mô hình mặc định nếu không có đường dẫn
        model = pipeline('text-classification',
                         model=model_path, return_all_scores=True) # Tạo pipeline phân loại văn bản
        return model

    def train_model(self, train_data, test_data, class_weights):
        if not self.model_path:
            self.model_path = "../trained_model"  # Thư mục mặc định nếu model_path rỗng

        os.makedirs(self.model_path, exist_ok=True)  # Đảm bảo thư mục tồn tại
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                                   num_labels=self.num_labels,
                                                                   id2label=self.label_dict) # Tải mô hình để huấn luyện
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer) # Tạo collator để đệm dữ liệu

        print(f"Đường dẫn mô hình: {self.model_path}")
        training_args = TrainingArguments(
            output_dir=self.model_path,
            learning_rate=2e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            push_to_hub=True,
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.set_device(self.device) # Đặt thiết bị cho trainer
        trainer.set_class_weights(class_weights) # Đặt trọng số lớp

        trainer.train() # Huấn luyện mô hình

        # Giải phóng bộ nhớ
        del trainer, model
        gc.collect()

        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def simplify_jutsu(self, jutsu):
        # Đơn giản hóa nhãn jutsu
        if "Genjutsu" in jutsu:
            return "Genjutsu"
        if "Ninjutsu" in jutsu:
            return "Ninjutsu"
        if "Taijutsu" in jutsu:
            return "Taijutsu"

    def preprocess_function(self, tokenizer, examples):
        # Tiền xử lý dữ liệu với tokenizer
        return tokenizer(examples['text_cleaned'], truncation=True)

    def load_data(self, data_path):
        # Kiểm tra xem tệp có tồn tại không
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Tệp tại {data_path} không tồn tại.")

        # Kiểm tra xem tệp có rỗng không
        if os.path.getsize(data_path) == 0:
            raise ValueError(f"Tệp tại {data_path} rỗng.")

        try:
            # Thử tải định dạng JSON Lines
            df = pd.read_json(data_path, lines=True)
        except ValueError as e:
            # Nếu thất bại, thử định dạng JSON thông thường
            try:
                df = pd.read_json(data_path)
            except ValueError:
                raise ValueError(
                    f"Không thể phân tích {data_path}. Hãy đảm bảo đó là tệp JSON hoặc JSON Lines hợp lệ. Lỗi: {str(e)}")

        # Kiểm tra xem dataframe có rỗng không
        if df.empty:
            raise ValueError(f"Không tải được dữ liệu từ {data_path}. Tệp có thể rỗng hoặc không đúng định dạng.")

        # Xử lý dữ liệu
        df['jutsu_type_simplified'] = df['jutsu_type'].apply(self.simplify_jutsu)
        df['text'] = df['jutsu_name'] + ". " + df['jutsu_description']
        df[self.label_column_name] = df['jutsu_type_simplified']
        df = df[['text', self.label_column_name]]
        df = df.dropna()

        # Làm sạch văn bản
        cleaner = Cleaner()
        df['text_cleaned'] = df[self.text_column_name].apply(cleaner.clean)

        # Mã hóa nhãn
        le = preprocessing.LabelEncoder()
        le.fit(df[self.label_column_name].tolist())

        label_dict = {index: label_name for index, label_name in enumerate(le.classes_.tolist())}
        self.label_dict = label_dict
        df['label'] = le.transform(df[self.label_column_name].tolist())

        # Chia tập huấn luyện và kiểm tra
        df_train, df_test = train_test_split(df,
                                             test_size=self.test_size,
                                             stratify=df['label'])

        # Chuyển từ Pandas sang Dataset của Hugging Face
        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)

        # Mã hóa dữ liệu
        tokenized_train = train_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples),
                                            batched=True)
        tokenized_test = test_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples),
                                          batched=True)

        return tokenized_train, tokenized_test

    def load_tokenizer(self):
        # Tải tokenizer từ model_path nếu tồn tại, nếu không thì từ model_name
        if self.model_path and huggingface_hub.repo_exists(self.model_path):
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer

    def postprocess(self, model_output):
        # Xử lý đầu ra mô hình
        output = []
        for pred in model_output:
            label = max(pred, key=lambda x: x['score'])['label']
            output.append(label)
        return output

    def classify_jutsu(self, text):
        # Phân loại văn bản
        model_output = self.model(text)
        predictions = self.postprocess(model_output)
        return predictions