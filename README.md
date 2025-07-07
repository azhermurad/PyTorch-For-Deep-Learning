# 🧠 Learn PyTorch for Deep Learning 

The course follows a **code-first, hands-on approach** and covers everything from PyTorch fundamentals to deploying real-world models.

---

## 📚 Course Curriculum Overview

| Module | Description |
|--------|-------------|
| `00` | PyTorch Fundamentals (Tensors, Autograd, Devices) |
| `01` | Deep Learning Workflow (End-to-end model training) |
| `02` | Neural Network Classification (Custom models with `nn.Module`) |
| `03` | Computer Vision (Image classification with CNNs) |
| `04` | Custom Datasets and Transforms |
| `05` | Going Modular (Reusable model/train/eval modules) |
| `06` | Transfer Learning (Fine-tuning pretrained models) |
| `07` | Experiment Tracking (with Weights & Biases) |
| `08` | Paper Replication (Food101 subset) |
| `09` | Model Deployment (Gradio + Hugging Face Spaces) |

---

## 🔍 Model Summaries and Architectures

### 1. 🧱 PyTorch Fundamentals
- Explored: `torch.tensor`, `autograd`, GPU acceleration (`torch.device`)
- ✅ Built small tensor-based neural networks manually (no `nn.Module`)
- 🧠 Learned backpropagation via `loss.backward()` and `.grad` tracking

### 2. 🔁 End-to-End Workflow Model
- Dataset: Binary classification (e.g., moon dataset or synthetic)
- Model: `nn.Sequential` with `Linear`, `ReLU`, `Sigmoid`
- Optimizer: `torch.optim.SGD`
- Loss: `torch.nn.BCELoss`
- Outcome: Built and trained a complete model pipeline from scratch

### 3. 🧠 Neural Network Classification
- Dataset: FashionMNIST (10-class grayscale images)
- Architecture:
  ```python
  class FashionMNISTModel(nn.Module):
      def __init__(self):
          super().__init__()
          self.flatten = nn.Flatten()
          self.layer_stack = nn.Sequential(
              nn.Linear(28*28, 128),
              nn.ReLU(),
              nn.Linear(128, 64),
              nn.ReLU(),
              nn.Linear(64, 10)
          )
      def forward(self, x): return self.layer_stack(self.flatten(x))
  ```
- Metrics: Accuracy, precision, recall using `torchmetrics`

### 4. 🖼️ Computer Vision CNN
- Dataset: CIFAR-10 or FoodVisionMini (custom)
- Architecture: Custom Convolutional Neural Network
  ```python
  class TinyCNN(nn.Module):
      def __init__(self):
          super().__init__()
          self.conv_block = nn.Sequential(
              nn.Conv2d(3, 32, 3),
              nn.ReLU(),
              nn.MaxPool2d(2),
              nn.Conv2d(32, 64, 3),
              nn.ReLU(),
              nn.MaxPool2d(2),
          )
          self.classifier = nn.Sequential(
              nn.Flatten(),
              nn.Linear(64*6*6, 128),
              nn.ReLU(),
              nn.Linear(128, 10)
          )
      def forward(self, x): return self.classifier(self.conv_block(x))
  ```

### 5. 🧩 Custom Datasets & Transforms
- Built a PyTorch `Dataset` and `DataLoader` for image folders
- Applied transforms like `Resize`, `RandomCrop`, `ToTensor`, `Normalize`
- Learned how to visualize transformed datasets and debug data pipelines

### 6. 🏗️ Going Modular
- Refactored the pipeline into:
  - `engine.py` → training/validation loops
  - `model_builder.py` → model definitions
  - `data_setup.py` → dataloaders/transforms
  - `utils.py` → helper functions (plotting, saving, accuracy)
- ✅ More readable, reusable, production-ready code

### 7. 🔁 Transfer Learning
- Dataset: Food101 (10-class subset)
- Base Model: `EfficientNetB0` from `torchvision.models`
- Strategy: 
  - Freeze base layers
  - Replace classifier head with new `nn.Linear`
  - Fine-tune with lower learning rate
- Optimizer: `Adam`
- ✅ Significantly improved training time and accuracy

### 8. 📄 Paper Replicating – FoodVision Mini
- Task: Classify images into 3 food classes
- Architecture: Transfer Learning with `EfficientNetB2`
- Logging: Tracked loss, accuracy, learning rate using `Weights & Biases`
- Result: Achieved >90% accuracy within few epochs

### 9. 🌍 Model Deployment
- Tools: `Gradio` for web UI, `torchvision.transforms` for preprocessing
- UI: Upload food image → Predict top class
- Hosted on: [Hugging Face Spaces](https://huggingface.co/spaces)
- ✅ Real-time prediction with uploaded image

---

## 🧪 Tools & Libraries Used

- `PyTorch`, `torchvision`, `torchmetrics`
- `matplotlib`, `pandas`, `numpy`
- `scikit-learn`, `wandb`, `gradio`

---

## 🏆 Projects

| Project | Description | Tech |
|--------|-------------|------|
| `FoodVision Mini` | Image classifier with 3 food classes | CNN / Transfer Learning |
| `Food101 (10-class)` | Fine-tuned EfficientNetB2 | Transfer Learning |
| `FoodVision Gradio` | Web demo of food classifier | Gradio + Hugging Face |

---

## 📂 Repository Structure

```bash
.
├── 00_pytorch_fundamentals/
├── 01_pytorch_workflow/
├── 02_neural_network_classification/
├── 03_computer_vision/
├── 04_custom_datasets/
├── 05_going_modular/
├── 06_transfer_learning/
├── 07_experiment_tracking/
├── 08_paper_replicating/
├── 09_model_deployment/
├── extras/
└── README.md
```

