import html
import json
import pickle
import re

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------------------------------------------------
# Files saved by the notebook (keep them in the same folder as app.py)
# ----------------------------------------------------------------------
MODEL_PATH = "next_word_lstm.keras"   # change to "next_word_lstm_v1.keras" if you saved it under that name
TOKENIZER_PATH = "tokenizer.pkl"
CONFIG_PATH = "config.json"

# ----------------------------------------------------------------------
# Generation settings (tune these to change the style of the output)
# ----------------------------------------------------------------------
TEMPERATURE = 0.7   # lower = safer and more predictable, higher = more surprising (1.0 = no change)
TOP_K = 40          # each step only chooses among the K most likely words

CHAT_HEIGHT = 450   # height (px) of the scrolling chat box


# ----------------------------------------------------------------------
# Text cleaning: must be IDENTICAL to the notebook's clean_text
# ----------------------------------------------------------------------
def _join_abbrev(m):
    return re.sub(r"[\s.]", "", m.group(0))


def clean_text(t):
    t = t.lower()
    t = t.replace("’", "'").replace("‘", "'")
    t = t.replace("“", "").replace("”", "").replace('"', "")
    t = re.sub(r"\b[a-z](?: \. [a-z])+\b", _join_abbrev, t)   # "o . w . l . s" -> "owls"
    t = re.sub(r"(?<=[a-z0-9])\s*'\s*s\b", "'s", t)           # "hermione ' s"  -> "hermione's"
    t = re.sub(r"[^a-z0-9'\s]", " ", t)                       # drop punctuation, keep apostrophes
    t = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", " ", t)       # drop apostrophes used as quote marks
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ----------------------------------------------------------------------
# Load model, tokenizer and config once
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    return model, tokenizer, config


# ----------------------------------------------------------------------
# Sentence generation
# ----------------------------------------------------------------------
def pick_next_index(probs, last_index, rng, temperature=TEMPERATURE, top_k=TOP_K):
    """Choose the next word index from the model's probabilities.

    1. block padding, <OOV> and the word that was just written (stops "the the the")
    2. temperature: sharpen the distribution so likely words win more often
    3. top-k: throw away everything except the k most likely words
    4. sample randomly from what is left, weighted by probability
    """
    probs = probs.copy()
    probs[0] = 0.0
    probs[last_index] = 0.0

    probs = probs ** (1.0 / temperature)

    k = min(top_k, len(probs))
    keep = np.argpartition(probs, -k)[-k:]
    filtered = np.zeros_like(probs)
    filtered[keep] = probs[keep]

    filtered /= filtered.sum()
    return int(rng.choice(len(filtered), p=filtered))


def generate_words(text, n_words, model, tokenizer, config, rng):
    """Return a list of n_words generated words, or None if no input word is recognised."""
    input_len = config["input_len"]
    oov_index = config["oov_index"]

    seq = tokenizer.texts_to_sequences([clean_text(text)])[0]
    if not any(i != oov_index for i in seq):
        return None

    words = []
    for _ in range(n_words):
        padded = pad_sequences([seq[-input_len:]], maxlen=input_len,
                               padding="pre", truncating="pre")
        probs = model(padded, training=False).numpy()[0].astype("float64")
        probs[oov_index] = 0.0                    # never produce <OOV>
        nxt = pick_next_index(probs, seq[-1], rng)
        seq.append(nxt)                           # the new word becomes part of the context
        words.append(tokenizer.index_word.get(nxt, "?"))
    return words


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------
st.set_page_config(page_title="Harry Potter Sentence Generator", page_icon="⚡")

# style for the highlighted (generated) words
st.markdown(
    """
    <style>
    mark.generated {
        background-color: rgba(255, 193, 7, 0.40);
        color: inherit;
        padding: 0 0.2em;
        border-radius: 0.25em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    model, tokenizer, config = load_artifacts()
except Exception as e:
    st.error(f"Could not load the model files: {e}")
    st.stop()

# header: title, description, slider
st.title("⚡ Harry Potter Next Word Predictor")
st.write(
    "An LSTM neural network trained on all seven Harry Potter books. "
    "Type the first few words, choose how many words to add, and it writes the rest of the sentence."
)
num_words = st.slider("Number of words to generate", min_value=1, max_value=100, value=10)

# chat history is kept between reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# scrolling chat box, then the input underneath it
chat_box = st.container(height=CHAT_HEIGHT)
user_text = st.chat_input("Type the first words of a sentence, e.g. 'harry looked at the'")

with chat_box:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    if user_text:
        shown_input = html.escape(user_text.strip())

        with st.chat_message("user"):
            st.markdown(shown_input)
        st.session_state.messages.append({"role": "user", "content": shown_input})

        with st.spinner("Writing..."):
            words = generate_words(user_text, num_words, model, tokenizer, config,
                                   np.random.default_rng())

        if words is None:
            reply = "None of those words are in my vocabulary. Try a few common words from the books."
        else:
            reply = f'{shown_input} <mark class="generated">{" ".join(words)}</mark>'

        with st.chat_message("assistant"):
            st.markdown(reply, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": reply})