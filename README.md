# Predicting the Next Word — NLP with Shakespeare’s Sonnets

This repository contains a compact, hands-on notebook that demonstrates how to build and train a word prediction model using Shakespeare’s Sonnets. It covers text preprocessing, tokenization, n-gram generation, sequence padding, model creation using a Bidirectional LSTM, and text generation.

---

## What’s in this repo

**Predicting_the_next_word.ipynb** — an end-to-end notebook that includes:
- Text preprocessing and tokenization with Keras
- Generation of n-gram sequences
- Sequence padding and one-hot encoding
- Model creation with Embedding and Bidirectional LSTM layers
- Model training and visualization of accuracy and loss
- Text generation using the trained model

---

## Dataset

**Source:** Shakespeare Sonnets Dataset  
Contains more than 2000 lines of text from Shakespeare’s sonnets, downloaded using:

```bash
!gdown --id 108jAePKK4R3BVYBbYJZ32JWUwxeMg20K
```

Each line is treated as a separate training sequence.

---

## Workflow Overview

| Step | Description |
|------|--------------|
| 1. Load & preprocess data | Lowercase text, split into lines, and tokenize using Keras `Tokenizer`. |
| 2. Generate n-grams | Build all partial sequences to predict the next word. |
| 3. Pad sequences | Normalize sequence lengths using `pad_sequences`. |
| 4. Split features & labels | Create predictors (`X`) and one-hot encoded targets (`y`). |
| 5. Build model | Use an Embedding + Bidirectional LSTM + Dense(softmax) architecture. |
| 6. Train | Train with categorical crossentropy and Adam optimizer for 50 epochs. |
| 7. Evaluate | Plot loss and accuracy curves. |
| 8. Generate text | Predict next words based on a seed phrase. |

---

## Model Architecture

```
Embedding(total_words, 100, input_length=max_sequence_len-1)
→ Bidirectional(LSTM(150))
→ Dense(total_words, activation='softmax')
```

**Training performance:**
- Accuracy: approximately 84–85%
- Loss: decreases steadily to around 0.5–0.6

**Training curves:**

| Training Accuracy | Training Loss |
|--------------------|----------------|
| ![Training accuracy](./images/training_accuracy.png) | ![Training loss](./images/training_loss.png) |

(See the `/images` folder or notebook output for the plots.)

---

## Example: Model in Action

```python
seed_text = "Help me Obi Wan Kenobi, you're my only hope"
next_words = 100
```

**Generated text:**

> then my argument more too still to die to prove each wit forth loss of thee do have more more new still new spent fickle worth poison dumb that best stol'n is it contains friend still me untrue...

---

## Highlights and Key Takeaways

- Demonstrates a word-level text generation pipeline using recurrent networks.
- Walks through essential NLP preprocessing steps: tokenization, padding, one-hot encoding.
- Trains a Bidirectional LSTM language model on a literary dataset.
- Includes visual training diagnostics for model performance.
- Easily adaptable for other text datasets or larger architectures.

---

## Quick Start

### 1. Environment setup
Recommended: Python 3.8+ (3.10+ suggested)

```bash
# create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate     # Windows
# or
source .venv/bin/activate      # macOS/Linux

# install dependencies
pip install --upgrade pip
pip install numpy matplotlib tensorflow keras gdown jupyterlab
```

### 2. Run the notebook
```bash
jupyter lab
```
Open `Predicting_the_next_word.ipynb` and execute cells sequentially.

---

## Outputs

- Trained model weights (H5 or TensorFlow SavedModel format)
- Tokenizer object for prediction
- Accuracy and loss plots
- Generated text predictions

---

## License and Contact

MIT License — maintained by PXDHU.  
Issues, discussions, and pull requests are welcome.
