# 🧠 Sentiment Analysis & Mental Health Classifier

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%96-HuggingFace-green)](https://huggingface.co/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10S50rldkYKXEDHM15sVVdah7K19cVLaE?usp=sharing)

A **multi-task learning model** using **DistilBERT** for **binary sentiment analysis** (negative/positive) and **multi-class mental health classification** (7 clinical categories). Optimized for GPU with Automatic Mixed Precision (AMP) training.

## ✨ Features

- 🚀 **Multi-Task Learning**: Single model predicts both sentiment (binary) and mental health status (7 classes)
- ⚡ **Production-Ready**: AMP, optimized DataLoader (`num_workers=2`, `pin_memory`), gradient accumulation
- 📊 **Rich Evaluation**: Confusion matrices, ROC curves, class-wise AUC
- 🔄 **Data Sources**: Kaggle mental health dataset + [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) (15K samples)
- 🎯 **Clinical Accuracy**: Trained 30 epochs, saves best model by validation loss
- 💻 **Colab-Native**: Google Drive integration, one-click GPU setup

## 📈 Model Architecture

```
DistilBERT (Pre-trained)
    ↓ (CLS token pooling + Dropout 0.2)
├── Sentiment Head: Linear(768 → 2)  [Neg/Pos]
└── MH Head:      Linear(768 → 7)    [Normal, Depression, Suicidal, Anxiety, Stress, Bi-Polar, Personality Disorder]
```

**Training Config:**
| Parameter | Value |
|-----------|-------|
| Epochs | 30 |
| Max Length | 256 |
| Batch Size | 16 |
| Learning Rate | 1e-5 |
| Optimizer | AdamW |

## 🚀 Quick Start (Google Colab)

1. Open [Colab notebook](https://colab.research.google.com/drive/10S50rldkYKXEDHM15sVVdah7K19cVLaE?usp=sharing)
2. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Upload `CombinedData.csv` to `/content/drive/MyDrive/`
4. **Runtime → Change Runtime Type → GPU**
5. Run all cells → **Model trains automatically!**

```bash
# After training, best model saves as:
multi_task_best_model.pth
```

## 📱 Screenshots

### 🏥 Mental Health Confusion Matrix

![MH Confusion Matrix](screenshots/mh_confusion_matrix.png)
_Heatmap showing clinical classification performance across 7 mental health categories_

### 😊 Sentiment Confusion Matrix

![Sentiment Confusion Matrix](screenshots/sentiment_confusion_matrix.png)
_Binary classification: Negative vs Positive sentiment_

### 📊 ROC Curves (Clinical Separability)

![ROC Curves](screenshots/roc_curves.png)
_Multi-class ROC showing per-category AUC scores_

> **💡 Tip**: Run the notebook locally, take screenshots of the plots, create `/screenshots/` folder, and replace placeholder links!

## 🎯 Performance Highlights

- **Dual-Head Architecture**: Shared BERT backbone + task-specific classifiers
- **Data Augmentation**: 15K GoEmotions (negative-focused) + Kaggle clinical data
- **Robust Training**: Ignores dummy labels (`ignore_index=-1`), AMP for speed/memory
- **Model Checkpointing**: Auto-saves `multi_task_best_model.pth` on val loss improvement

## 🔮 Inference Example

```python
model = MultiTaskModel(num_mh=7, num_sent=2)
model.load_state_dict(torch.load('multi_task_best_model.pth'))
model.eval()

# Predict single text
text = "I feel completely hopeless and want to give up"
inputs = tokenizer(text, return_tensors='pt', max_length=256, truncation=True)
with torch.no_grad():
    mh_logits, sent_logits = model(**inputs)
    mh_pred = torch.argmax(mh_logits, dim=1)  # e.g., Depression (1)
    sent_pred = torch.argmax(sent_logits, dim=1)  # e.g., Negative (0)
```

## 🛠️ Local Setup

```bash
# Clone & Install
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-model.git
cd sentiment-analysis-model

pip install torch transformers datasets scikit-learn pandas tqdm seaborn matplotlib

# Run notebook (requires GPU)
jupyter notebook sentiment_analysis_model.ipynb
```

**Requirements:** `pip install -r requirements.txt`

## 🔮 Future Work

- [ ] Add Streamlit/Gradio demo app
- [ ] Deploy to Hugging Face Spaces
- [ ] LoRA fine-tuning for efficiency
- [ ] More datasets (Twitter, Reddit mental health)
- [ ] Explainability (SHAP/LIME)

## 🤝 Contributing

1. Fork the repo
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/)
- [PyTorch AMP](https://pytorch.org/docs/stable/amp.html)
- [GoEmotions Dataset](https://huggingface.co/datasets/google-research-datasets/go_emotions)
- Kaggle Mental Health community datasets

---
