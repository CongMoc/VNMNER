# run_convert_test.py
# Quick test to exercise convert_mm_examples_to_features with a dummy tokenizer
import traceback
try:
    from modules.datasets import dataset_roberta_main as drm
    from modules.datasets.dataset_roberta_main import MNERProcessor, convert_mm_examples_to_features
    import torch

    class DummyTokenizer:
        def tokenize(self, w):
            # simple tokenization: return the word itself (no subtokenization)
            return [w]
        def convert_tokens_to_ids(self, tokens):
            return [1] * len(tokens)

    # monkeypatch image_process to avoid opening actual image files
    def fake_image_process(path, transform):
        # return a tensor that matches expected image shape
        return torch.zeros(3, 224, 224)

    drm.image_process = fake_image_process

    processor = MNERProcessor()
    examples = processor.get_train_examples('sample_data')
    print('Loaded train examples:', len(examples))
    label_list = ["O","B-ORG","B-MISC","I-PER","I-ORG","B-LOC","I-MISC","I-LOC","B-PER","X","<s>","</s>"]
    auxlabel_list = ["O", "B", "I", "X", "<s>", "</s>"]

    features = convert_mm_examples_to_features(examples, label_list, auxlabel_list, 128, DummyTokenizer(), 224, 'sample_data')
    print('Converted features:', len(features))
except Exception as e:
    print('Exception during test:')
    traceback.print_exc()
