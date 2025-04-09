import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Dữ liệu test thực tế
real_text = ["This product is amazing, I am very satisfied with the quality and service."]

# Cấu hình dùng chung
MAX_WORDS = 10000
MAX_LEN = 200

def preprocess(texts, tokenizer):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN)
    return padded

def test_model(name, model_path, dataset_name, text_field):
    print(f"\n========== 🧪 ĐÁNH GIÁ MÔ HÌNH: {name} ==========")

    # Load mô hình
    model = tf.keras.models.load_model(model_path)

    # Load tập dữ liệu
    dataset = load_dataset(dataset_name)
    test_texts = dataset["test"][text_field][:1000]
    test_labels = dataset["test"]["label"][:1000]

    # Tạo tokenizer và chuẩn hóa văn bản
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(test_texts)
    test_padded = preprocess(test_texts, tokenizer)

    # Dự đoán trên tập test
    pred_probs = model.predict(test_padded)
    preds = (pred_probs > 0.5).astype("int32").flatten()

    # Đánh giá model
    print(classification_report(test_labels, preds))

    # Dự đoán sentiment cho văn bản thực tế
    real_input = preprocess(real_text, tokenizer)
    real_pred = model.predict(real_input)[0][0]
    sentiment = "Tích cực 👍" if real_pred > 0.5 else "Tiêu cực 👎"
    print(f"\n📢 Dự đoán sentiment cho văn bản thật:\n \"{real_text[0]}\"\n => {sentiment} (score: {real_pred:.2f})")


# Test 3 mô hình với các dataset tương ứng
test_model(
    name="IMDB (Phim)",
    model_path="sentiment_model_imdb.h5",
    dataset_name="imdb",
    text_field="text"
)

test_model(
    name="Amazon Polarity (Sản phẩm)",
    model_path="sentiment_model_amazon.h5",
    dataset_name="amazon_polarity",
    text_field="content"  # Chú ý: amazon dùng 'content' thay vì 'text'
)

test_model(
    name="Yelp Polarity (Nhà hàng)",
    model_path="sentiment_model_yelp.h5",
    dataset_name="yelp_polarity",
    text_field="text"
)
