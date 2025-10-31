#!/usr/bin/env python3
import os
from modules.datasets import dataset_roberta_main as drm
import re
from pathlib import Path

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

# Additional checks: look for suspicious labels that look like image ids or file names,
# and detect token/label length mismatches. Log findings to bad_label_lines.txt.
bad_log = Path('bad_label_lines.txt')
if bad_log.exists():
    bad_log.unlink()

suspect_pattern = re.compile(r'.*_img_.*', flags=re.IGNORECASE)
ext_pattern = re.compile(r'.*\.(jpg|jpeg|png|bmp)$', flags=re.IGNORECASE)

print('\n-- Scanning sbreadfile output for suspicious labels / mismatches --', flush=True)
bad_count = 0
for idx, (tokens, labels) in enumerate(data):
    # length mismatch
    if len(tokens) != len(labels):
        bad_count += 1
        msg = f"MISMATCH sample#{idx} img={imgs[idx]} tokens={len(tokens)} labels={len(labels)}\n"
        print(msg, end='', flush=True)
        bad_log.write_text(bad_log.read_text(encoding='utf-8') + msg if bad_log.exists() else msg, encoding='utf-8')

    # suspicious label values
    for j, lab in enumerate(labels):
        if lab is None:
            continue
        if suspect_pattern.match(lab) or ext_pattern.match(lab) or lab.startswith('IMGID:'):
            bad_count += 1
            context = ' '.join(tokens[max(0, j-5):j+6])
            msg = (f"SUSPECT_LABEL sample#{idx} token_index={j} img={imgs[idx]} label='{lab}' "
                   f"context='{context}'\n")
            print(msg, end='', flush=True)
            bad_log.write_text(bad_log.read_text(encoding='utf-8') + msg if bad_log.exists() else msg, encoding='utf-8')

print(f"Scan finished. suspicious/mismatch count: {bad_count}", flush=True)

# Test processor-level API
print("\n-- Calling MNERProcessor.get_train_examples --", flush=True)
proc = drm.MNERProcessor()
examples = proc.get_train_examples(data_dir)
print(f"get_train_examples -> examples: {len(examples)}", flush=True)
for i, ex in enumerate(examples[:3]):
    print(
        f"example#{i} guid={ex.guid} img_id={ex.img_id} text_a_preview={' '.join(ex.text_a.split()[:30])}", flush=True)

# Try to run convert_mm_examples_to_features on a small subset if transformers is available.
try:
    from transformers import AutoTokenizer
    print('\n-- Attempting to convert a small number of examples to features --', flush=True)
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
    label_list = proc.get_labels()
    auxlabel_list = proc.get_auxlabels()
    # Use only first 10 examples to keep it light
    subset = examples[:10]
    try:
        feats = drm.convert_mm_examples_to_features(subset, label_list, auxlabel_list, 256, tokenizer, 224,
                                                    os.path.join(data_dir, 'ner_image'))
        print(f"convert_mm_examples_to_features -> produced {len(feats)} feature objects", flush=True)
    except Exception as e:
        print(f"convert_mm_examples_to_features raised an exception: {e}", flush=True)
except Exception as e:
    print('transformers not available or tokenizer load failed:', e, flush=True)

print("\nTest finished.", flush=True)
