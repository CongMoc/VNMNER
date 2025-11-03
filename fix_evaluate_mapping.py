#!/usr/bin/env python3
"""
Script to fix evaluate() mapping in all training scripts.
Changes reverse_label_map (label->idx) to idx_to_label (idx->label).
"""

import os
import re

REPLACEMENT_CODE_DEV = """
            # Build idx->label mapping (needed by evaluate function)
            # Collect all label IDs present in predictions/ground truth
            all_ids = set(
                [lab_id for seq in (y_true_idx + y_pred_idx) for lab_id in seq])
            idx_to_label = {}
            # Map known labels from label_list (enumerate starting at 1)
            for i, label in enumerate(label_list, 1):
                idx_to_label[i] = label
            # Ensure pad/zero maps to 'O' or '<pad>' if present
            idx_to_label[0] = idx_to_label.get(0, 'O')
            # Cover any remaining IDs found in data by assigning 'O'
            for id_ in all_ids:
                if id_ not in idx_to_label:
                    idx_to_label[id_] = 'O'

            acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list,
                                     idx_to_label)
"""

REPLACEMENT_CODE_TEST = """
    # Build idx->label mapping for test evaluation (same as dev)
    all_ids = set(
        [lab_id for seq in (y_true_idx + y_pred_idx) for lab_id in seq])
    idx_to_label = {}
    for i, label in enumerate(label_list, 1):
        idx_to_label[i] = label
    idx_to_label[0] = idx_to_label.get(0, 'O')
    for id_ in all_ids:
        if id_ not in idx_to_label:
            idx_to_label[id_] = 'O'

    acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, idx_to_label)
"""

def fix_file(filepath):
    """Fix evaluate mapping in a single file."""
    if not os.path.exists(filepath):
        print(f"⏭️  Skip {filepath} (not found)")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes = []
    
    # Pattern 1: Dev evaluation with reverse_label_map
    pattern_dev = r'(\s+sentence_list\.append\(sentence\)\s*\n)\s+reverse_label_map\s*=\s*\{[^}]+\}\s*\n\s+acc,\s*f1,\s*p,\s*r\s*=\s*evaluate\([^)]+reverse_label_map\)'
    
    if re.search(pattern_dev, content):
        # Replace with idx_to_label mapping
        content = re.sub(
            pattern_dev,
            r'\1' + REPLACEMENT_CODE_DEV.strip(),
            content,
            count=1
        )
        changes.append("dev evaluation")
    
    # Pattern 2: Test evaluation with reverse_label_map  
    pattern_test = r'(\s+fout\.close\(\)\s*\n)\s+reverse_label_map\s*=\s*\{[^}]+\}\s*\n\s+acc,\s*f1,\s*p,\s*r\s*=\s*evaluate\([^)]+reverse_label_map\)'
    
    if re.search(pattern_test, content):
        content = re.sub(
            pattern_test,
            r'\1' + REPLACEMENT_CODE_TEST.strip(),
            content,
            count=1
        )
        changes.append("test evaluation")
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed {os.path.basename(filepath)}: {', '.join(changes)}")
        return True
    else:
        print(f"⏭️  Skip {os.path.basename(filepath)} (no changes needed)")
        return False

def main():
    """Fix all training scripts."""
    train_files = [
        "train_umt.py",
        "train_roberta_softmax_gate_multimodal.py",
        "train_roberta_crf_multimodal.py",
        "train_roberta_softmax_multimodal.py",
        "train_roberta_crf_gate_multimodal.py",
        "train_maf.py",
        "train_roberta_crf_gate_cl_multimodal.py",
        "train.py",
        "bert_train_umt.py",
        "bert_train_umt_pixelcnn_fixedlr.py",
        "train_umt_pixelcnn_external_context.py",
    ]
    
    print("="*70)
    print("🔧 Fixing evaluate() mapping: reverse_label_map → idx_to_label")
    print("="*70)
    print()
    
    fixed_count = 0
    for filename in train_files:
        if fix_file(filename):
            fixed_count += 1
    
    print()
    print("="*70)
    print(f"✨ Done! Fixed {fixed_count}/{len(train_files)} files")
    print("="*70)
    print()
    print("📋 Summary:")
    print("  • Changed reverse_label_map (label→idx) to idx_to_label (idx→label)")
    print("  • This matches the working wo_CL script")
    print("  • Now evaluate() will receive correct mapping format")
    print()

if __name__ == "__main__":
    main()
