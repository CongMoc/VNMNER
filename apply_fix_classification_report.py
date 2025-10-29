#!/usr/bin/env python3
"""
Script to fix ValueError: max() arg is an empty sequence in classification_report
This script patches train_umt_pixelcnn_fixedlr_wo_CL.py to handle empty predictions gracefully.

Usage:
    python apply_fix_classification_report.py
"""

import re
import os
import sys
from pathlib import Path


def is_already_patched(content):
    """Check if the file has already been patched."""
    # Check for key indicators that the fix has been applied
    indicators = [
        'if y_true and y_pred and len(y_true) > 0 and len(y_pred) > 0:',
        'Only add non-empty sequences',
        'Total predictions collected',
        'WARNING: No valid predictions found'
    ]

    found_count = sum(1 for indicator in indicators if indicator in content)

    # If we found at least 3 out of 4 indicators, file is already patched
    return found_count >= 3


def apply_fixes(filepath='train_umt_pixelcnn_fixedlr_wo_CL.py'):
    """Apply all necessary fixes to the training script."""

    if not os.path.exists(filepath):
        print(f"❌ Error: File '{filepath}' not found!")
        print(f"   Current directory: {os.getcwd()}")
        return False

    print(f"🔧 Reading file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already patched
    if is_already_patched(content):
        print("\n" + "="*70)
        print("✅ File is already patched!")
        print("="*70)
        print("\n📋 The file already contains all necessary fixes:")
        print("  • label_map.get() for safe dictionary access")
        print("  • Non-empty sequence checks before appending")
        print("  • Debug logging for prediction counts")
        print("  • Safety checks before classification_report()")
        print("  • Warning messages for empty predictions")
        print("\n✨ No changes needed - skipping patch application.")
        print("="*70)
        return True

    original_content = content
    fixes_applied = []

    # Fix 1: Use .get() instead of [] to avoid KeyError
    print("  📌 Fix 1: Replacing label_map[] with label_map.get()...")
    before_count = content.count('label_map[label_ids[i][j]]')
    content = content.replace(
        'if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "</s>"',
        'if label_map.get(label_ids[i][j], "UNK") != "X" and label_map.get(label_ids[i][j], "UNK") != "</s>"'
    )
    after_count = content.count('label_map[label_ids[i][j]]')
    if before_count > after_count:
        fixes_applied.append(
            f"Replaced {before_count - after_count} instances of label_map[] with label_map.get()")

    # Fix 2: Only add non-empty sequences in dev evaluation
    print("  📌 Fix 2: Adding check for non-empty sequences in dev evaluation...")
    # Look for the pattern where we append without checking
    dev_pattern = r'(else:\s+break)\s+y_true\.append\(temp_1\)\s+y_pred\.append\(temp_2\)\s+y_true_idx\.append\(tmp1_idx\)\s+y_pred_idx\.append\(tmp2_idx\)\s+(# Debug information|if y_true and y_pred)'

    if re.search(dev_pattern, content, re.DOTALL):
        dev_replacement = r'''\1
                # Only add non-empty sequences
                if temp_1 and temp_2:
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    y_true_idx.append(tmp1_idx)
                    y_pred_idx.append(tmp2_idx)

        \2'''
        content = re.sub(dev_pattern, dev_replacement,
                         content, flags=re.DOTALL, count=1)
        fixes_applied.append("Added non-empty check for dev evaluation")

    # Fix 3: Add debug info and safety check before classification_report (dev)
    print("  📌 Fix 3: Adding debug info and safety check for dev classification_report...")
    # Only add if not already present
    if 'Total predictions collected' not in content:
        dev_report_pattern = r'(y_pred_idx\.append\(tmp2_idx\))\s+report = classification_report'

        dev_report_replacement = r'''\1

        # Debug information
        logger.info(f"Total predictions collected: {len(y_pred)}")
        logger.info(f"Total true labels collected: {len(y_true)}")
        
        if y_true and y_pred and len(y_true) > 0 and len(y_pred) > 0:
            report = classification_report'''

        if re.search(dev_report_pattern, content, re.DOTALL):
            content = re.sub(
                dev_report_pattern, dev_report_replacement, content, flags=re.DOTALL, count=1)
            fixes_applied.append(
                "Added debug info and safety check for dev evaluation")

    # Fix 4: Add else clause for dev evaluation
    print("  📌 Fix 4: Adding warning message for empty dev predictions...")
    if 'WARNING: No valid predictions found in dev set' not in content:
        dev_else_pattern = r'(max_dev_f1 = F_score_dev\s+best_dev_epoch = train_idx)\s+(logger\.info\("\*+"\))'

        dev_else_replacement = r'''\1
        else:
            logger.warning("***** WARNING: No valid predictions found in dev set! *****")
            logger.warning(f"y_true length: {len(y_true)}, y_pred length: {len(y_pred)}")
            logger.warning("Skipping evaluation for this epoch.")

    \2'''

        if re.search(dev_else_pattern, content):
            content = re.sub(dev_else_pattern, dev_else_replacement, content)
            fixes_applied.append("Added warning for empty dev predictions")

    # Fix 5: Apply same fixes for test evaluation
    print("  📌 Fix 5: Adding check for non-empty sequences in test evaluation...")
    # Look for test evaluation pattern
    test_pattern = r'(break)\s+y_true\.append\(temp_1\)\s+y_pred\.append\(temp_2\)\s+y_true_idx\.append\(tmp1_idx\)\s+y_pred_idx\.append\(tmp2_idx\)\s+(# Debug information|logger\.info.*Total test|if y_true and y_pred)'

    if re.search(test_pattern, content, re.DOTALL):
        test_replacement = r'''\1
            # Only add non-empty sequences
            if temp_1 and temp_2:
                y_true.append(temp_1)
                y_pred.append(temp_2)
                y_true_idx.append(tmp1_idx)
                y_pred_idx.append(tmp2_idx)

    \2'''
        content = re.sub(test_pattern, test_replacement,
                         content, flags=re.DOTALL, count=1)
        fixes_applied.append("Added non-empty check for test evaluation")

    # Fix 6: Add test debug and safety check
    print("  📌 Fix 6: Adding debug info and safety check for test classification_report...")
    if 'Total test predictions collected' not in content:
        test_report_pattern = r'(y_pred_idx\.append\(tmp2_idx\))\s+report = classification_report\(y_true, y_pred, digits=4\)'

        test_report_replacement = r'''\1

    # Debug information
    logger.info(f"Total test predictions collected: {len(y_pred)}")
    logger.info(f"Total test true labels collected: {len(y_true)}")
    
    if y_true and y_pred and len(y_true) > 0 and len(y_pred) > 0:
        report = classification_report(y_true, y_pred, digits=4)'''

        if re.search(test_report_pattern, content, re.DOTALL):
            content = re.sub(test_report_pattern,
                             test_report_replacement, content, flags=re.DOTALL)
            fixes_applied.append(
                "Added debug info and safety check for test evaluation")

    # Fix 7: Add else clause for test evaluation
    print("  📌 Fix 7: Adding warning message for empty test predictions...")
    if 'WARNING: No valid predictions found in test set' not in content:
        # Only add the else clause if it doesn't already exist
        test_else_pattern = r'(writer\.write\("Overall: " \+ str\(p\) \+ \' \' \+\s+str\(r\) \+ \' \' \+ str\(f1\) \+ \'\\n\'\))(?!\s+else:)'

        test_else_replacement = r'''\1
    else:
        logger.warning("***** WARNING: No valid predictions found in test set! *****")
        logger.warning(f"y_true length: {len(y_true)}, y_pred length: {len(y_pred)}")
        logger.warning("Test evaluation skipped due to empty predictions.")'''

        if re.search(test_else_pattern, content, re.MULTILINE):
            content = re.sub(
                test_else_pattern, test_else_replacement, content, flags=re.MULTILINE)
            fixes_applied.append("Added warning for empty test predictions")

    # Check if any changes were made
    if content == original_content:
        print("\n⚠️  Warning: No changes were made. The file may already be patched or patterns not found.")
        return True

    # Write the fixed content back
    print(f"\n💾 Writing fixed content to {filepath}...")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    # Print summary
    print("\n" + "="*70)
    print("✅ Fix applied successfully!")
    print("="*70)
    print("\n📋 Changes made:")
    for i, fix in enumerate(fixes_applied, 1):
        print(f"  {i}. {fix}")

    print("\n🎯 What these fixes do:")
    print("  • Prevent KeyError by using label_map.get() with default value")
    print("  • Only add non-empty sequences to y_true/y_pred lists")
    print("  • Add debug logging to track prediction counts")
    print("  • Add safety checks before calling classification_report()")
    print("  • Show warnings instead of crashing when no predictions found")

    print("\n✨ The script should now run without 'empty sequence' errors!")
    print("="*70)

    return True


def main():
    """Main function."""
    print("="*70)
    print("🔧 Classification Report Empty Sequence Fix Script")
    print("="*70)
    print()

    # Try to find the file
    possible_paths = [
        'train_umt_pixelcnn_fixedlr_wo_CL.py',
        './train_umt_pixelcnn_fixedlr_wo_CL.py',
        '/kaggle/working/VNMNER/train_umt_pixelcnn_fixedlr_wo_CL.py',
    ]

    filepath = None
    for path in possible_paths:
        if os.path.exists(path):
            filepath = path
            break

    if filepath is None:
        # Check current directory
        cwd = os.getcwd()
        print(f"📂 Current directory: {cwd}")
        print(f"📄 Files in current directory:")
        for f in os.listdir(cwd):
            if f.endswith('.py'):
                print(f"   - {f}")
        print()
        print("❌ Could not find 'train_umt_pixelcnn_fixedlr_wo_CL.py'")
        print("   Please run this script from the VNMNER directory or specify the path.")
        return 1

    success = apply_fixes(filepath)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
