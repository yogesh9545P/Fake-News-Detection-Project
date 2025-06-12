
# 📰 Fake News Detection using LSTM

This project uses Natural Language Processing (NLP) and Deep Learning (LSTM) to classify news articles as **Fake** or **Real**.

## 📁 Dataset
- [Fake.csv](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [True.csv](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

## 🧠 Model
- Tokenizer with 10,000 words
- Padding up to 500 words
- LSTM layer with 64 units
- Output: Binary (Fake/Real)

## 📈 Training Performance
| Metric     | Value    |
|------------|----------|
| Accuracy   | ~90%     |
| Loss       | Low      |

(Insert your accuracy_plot.png and loss_plot.png here)

## 🛠 Setup
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
python fake_news_model.py
```

## 📦 Output
- Trained model: `fake_news_model.h5`
- Accuracy & loss plots
- Classification report

## 🔗 LinkedIn Hashtags
#DataScience #FakeNewsDetection #DeepLearning #NLP #PythonProject
