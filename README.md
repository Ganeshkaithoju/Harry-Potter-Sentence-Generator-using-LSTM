# ⚡ Harry Potter Sentence Generator

Type the first few words of a sentence, pick how many words to add, and an LSTM neural network writes the rest in the style of the Harry Potter books.

> An unofficial fan project for learning. The book text is not included in this repo.

## How it works

The model reads a sentence word by word and predicts the next word. The app repeats this: it picks a word, adds it to the sentence, and predicts again until it has the number of words you asked for.

- **Data:** all seven books, about 73,000 cleaned sentences
- **Model:** word embeddings (100 dims) → 2 stacked LSTM layers (192 units each) → softmax over an 8,000-word vocabulary
- **Built with:** TensorFlow / Keras, pandas, NumPy, scikit-learn, Streamlit

## Results

| Metric | Value |
|---|---|
| Validation perplexity | about 148 |
| Top-1 accuracy | roughly 18% |
| Top-5 accuracy | roughly 36% |

These are normal numbers for a word-level model on fiction, where many different words can fit the next slot.

## Project files

| File | What it is |
|---|---|
| `NW_pred_project.ipynb` | Colab notebook: cleaning, training and evaluation |
| `app.py` | Streamlit app |
| `next_word_lstm.keras` | Trained model |
| `tokenizer.pkl` | Word-to-number mapping from training |
| `config.json` | Input length, vocabulary size, `<OOV>` index |

## Run it

```bash
pip install streamlit tensorflow numpy
streamlit run app.py
```

Use the same TensorFlow version you trained with, otherwise the model may not load. Keep the model, tokenizer and config files in the same folder as `app.py`.

## Using the app

1. Set **Number of words to generate** with the slider (1 to 100).
2. Type the start of a sentence in the chat box.
3. Your words appear with the generated words highlighted in yellow.

Each word is chosen at random from the 40 most likely options, so the same input gives a different sentence each time. You can change the style at the top of `app.py`:

- `TEMPERATURE`: lower is safer and more predictable, higher is more surprising
- `TOP_K`: how many candidate words are considered at each step
- `CHAT_HEIGHT`: height of the chat box in pixels

## Limitations

- The model only looks back 29 words, so sentences read like the books but don't follow a plot.
- Punctuation was removed during training, so the output has none and always has exactly the number of words you ask for.
- The vocabulary is capped at 8,000 words, so rare names and spells are never generated.
- Words outside the vocabulary are ignored in your input.

## Ideas for next steps

- Pretrained embeddings (GloVe)
- A GRU or a larger LSTM
- Keeping punctuation so the model can learn where sentences end