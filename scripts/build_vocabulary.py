"""
WildMatch AI - Training Vocabulary Construction Script
Constructs and validates the word-level Vocabulary using ONLY the training captions split.
"""

import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from preprocessing.text_preprocessing import (
    Vocabulary,
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_IDX,
    UNK_IDX,
    BOS_IDX,
    EOS_IDX,
    MAX_SEQ_LENGTH,
    MIN_VOCAB_FREQ,
)

TRAIN_CSV = "dataset/splits/train.csv"
VAL_CSV = "dataset/splits/validation.csv"
TEST_CSV = "dataset/splits/test.csv"
VOCAB_OUTPUT = "dataset/vocabulary.json"


def main():
    print("=== STEP 1: LOAD DATASET SPLITS ===")
    assert os.path.exists(TRAIN_CSV), f"Train CSV missing: {TRAIN_CSV}"
    assert os.path.exists(VAL_CSV), f"Validation CSV missing: {VAL_CSV}"
    assert os.path.exists(TEST_CSV), f"Test CSV missing: {TEST_CSV}"

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    assert len(train_df) == 140, f"Expected 140 train records, got {len(train_df)}"
    assert len(val_df) == 30, f"Expected 30 val records, got {len(val_df)}"
    assert len(test_df) == 30, f"Expected 30 test records, got {len(test_df)}"
    print(f"Loaded splits: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test.")

    train_token_lists = [json.loads(t) for t in train_df["tokens"]]
    val_token_lists = [json.loads(t) for t in val_df["tokens"]]
    test_token_lists = [json.loads(t) for t in test_df["tokens"]]

    print("\n=== STEP 2: BUILD VOCABULARY EXCLUSIVELY ON TRAINING TOKENS ===")
    vocab = Vocabulary(
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        max_seq_length=MAX_SEQ_LENGTH,
        min_freq=MIN_VOCAB_FREQ,
    )
    vocab.build_vocab(train_token_lists)

    print(f"Total vocabulary size: {len(vocab)}")
    num_special = len(vocab.special_tokens)
    num_words = len(vocab) - num_special
    print(f"Special tokens count: {num_special}")
    print(f"Unique training words: {num_words}")

    print("\n=== STEP 3: SAVE VOCABULARY JSON ===")
    vocab.save_json(VOCAB_OUTPUT)
    assert os.path.exists(VOCAB_OUTPUT), f"Failed to create {VOCAB_OUTPUT}"
    print(f"Saved vocabulary to {VOCAB_OUTPUT}")

    print("\n=== STEP 4: COMPREHENSIVE VALIDATION ===")
    # 1. Reload vocabulary
    vocab_reloaded = Vocabulary.load_json(VOCAB_OUTPUT)
    assert len(vocab_reloaded) == len(vocab), "Reloaded vocab size mismatch"
    assert vocab_reloaded.word2idx == vocab.word2idx, "Reloaded word2idx mismatch"
    assert vocab_reloaded.idx2word == vocab.idx2word, "Reloaded idx2word mismatch"

    # 2. Verify special token indices
    assert vocab.pad_idx == PAD_IDX == 0, f"Expected PAD_IDX 0, got {vocab.pad_idx}"
    assert vocab.unk_idx == UNK_IDX == 1, f"Expected UNK_IDX 1, got {vocab.unk_idx}"
    assert vocab.bos_idx == BOS_IDX == 2, f"Expected BOS_IDX 2, got {vocab.bos_idx}"
    assert vocab.eos_idx == EOS_IDX == 3, f"Expected EOS_IDX 3, got {vocab.eos_idx}"
    print("Special token indices verified: <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3.")

    # 3. Verify no duplicate entries in word2idx or idx2word
    assert len(vocab.word2idx) == len(set(vocab.word2idx.keys())), "Duplicate keys in word2idx"
    assert len(vocab.word2idx.values()) == len(set(vocab.word2idx.values())), "Duplicate values in word2idx"
    assert len(vocab.idx2word) == len(set(vocab.idx2word.keys())), "Duplicate keys in idx2word"
    assert len(vocab.idx2word.values()) == len(set(vocab.idx2word.values())), "Duplicate values in idx2word"
    print("No duplicates in word2idx or idx2word mappings.")

    # 4. Bidirectional mapping consistency
    for word, idx in vocab.word2idx.items():
        assert vocab.idx2word[idx] == word, f"Inconsistent mapping for {word} -> {idx}"
    for idx, word in vocab.idx2word.items():
        assert vocab.word2idx[word] == idx, f"Inconsistent mapping for {idx} -> {word}"
    print("Bidirectional token-to-index and index-to-token mappings are 100% consistent.")

    # 5. Verify all entries are derived from training captions only
    all_train_words = set(w for tl in train_token_lists for w in tl)
    assert num_words == len(all_train_words), f"Mismatch between training words ({len(all_train_words)}) and vocab words ({num_words})"
    vocab_word_set = set(vocab.word2idx.keys()) - set(vocab.special_tokens)
    assert vocab_word_set == all_train_words, "Vocabulary words differ from unique training words!"
    print("All vocabulary entries strictly derived from training captions only.")

    # 6. Verify every training token maps to its valid non-UNK index
    for tokens in train_token_lists:
        for t in tokens:
            idx = vocab[t]
            assert idx != vocab.unk_idx, f"Training token '{t}' mapped to UNK!"
            assert idx >= 4, f"Training token '{t}' mapped to special token index {idx}!"
    print("Every training token correctly maps to a valid non-UNK index.")

    # 7. Check validation and test split OOV tokens map to UNK
    val_tokens_flat = [w for tl in val_token_lists for w in tl]
    test_tokens_flat = [w for tl in test_token_lists for w in tl]

    val_unk_tokens = [w for w in val_tokens_flat if vocab[w] == vocab.unk_idx]
    test_unk_tokens = [w for w in test_tokens_flat if vocab[w] == vocab.unk_idx]

    val_oov_unique = sorted(set(val_unk_tokens))
    test_oov_unique = sorted(set(test_unk_tokens))

    print(f"\nValidation tokens mapped to <UNK>: {len(val_unk_tokens)} occurrences across {len(val_oov_unique)} unique words:")
    print(f"  {val_oov_unique}")
    print(f"Test tokens mapped to <UNK>: {len(test_unk_tokens)} occurrences across {len(test_oov_unique)} unique words:")
    print(f"  {test_oov_unique}")

    for w in val_oov_unique:
        assert w not in vocab, f"OOV word {w} unexpectedly in vocab"
        assert vocab[w] == vocab.unk_idx, f"OOV word {w} did not map to UNK"

    for w in test_oov_unique:
        assert w not in vocab, f"OOV word {w} unexpectedly in vocab"
        assert vocab[w] == vocab.unk_idx, f"OOV word {w} did not map to UNK"

    print("\nALL VOCABULARY VALIDATION CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
