#!/usr/bin/env python3
import os
from modules.datasets import dataset_roberta_main as drm

print("== Data reader test script ==", flush=True)
root = os.path.dirname(__file__)
data_dir = os.path.join(root, "sample_data")
train_path = os.path.join(data_dir, "train.txt")
print(f"Using train file: {train_path}", flush=True)

# Test low-level reader
print("\n-- Calling _read_sbtsv (sbreadfile) --", flush=True)
data, imgs, aux = drm._read_sbtsv(train_path)
print(f"sbreadfile -> samples: {len(data)}, images: {len(imgs)}", flush=True)
for i in range(min(3, len(data))):
    tokens, labels = data[i]
    print(
        f"sample#{i} img={imgs[i]} tokens_preview={tokens[:20]} labels_preview={labels[:20]}", flush=True)

# Test processor-level API
print("\n-- Calling MNERProcessor.get_train_examples --", flush=True)
proc = drm.MNERProcessor()
examples = proc.get_train_examples(data_dir)
print(f"get_train_examples -> examples: {len(examples)}", flush=True)
for i, ex in enumerate(examples[:3]):
    print(
        f"example#{i} guid={ex.guid} img_id={ex.img_id} text_a_preview={' '.join(ex.text_a.split()[:30])}", flush=True)

print("\nTest finished.", flush=True)
