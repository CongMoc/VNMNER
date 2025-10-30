from PIL import Image
from torchvision import transforms
import torch
import logging
import os
print(
    f"LOADED modules/datasets/dataset_roberta_main.py from {__file__}", flush=True)
logger = logging.getLogger(__name__)
SPECIAL_TOKENS = ['\ufe0f', '\u200d', '\u200b', '\x92']
URL_PREFIX = 'http'


class SBInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, img_id, label=None, auxlabel=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.img_id = img_id
        self.label = label
        # Please note that the auxlabel is not used in SB
        # it is just kept in order not to modify the original code
        self.auxlabel = auxlabel


class SBInputFeatures(object):
    """A single set of features of data"""

    def __init__(self, input_ids, input_mask, added_input_mask, segment_ids, img_feat, img_ti_feat, label_id, auxlabel_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.added_input_mask = added_input_mask
        self.segment_ids = segment_ids
        self.img_feat = img_feat
        self.img_ti_feat = img_ti_feat
        self.label_id = label_id
        self.auxlabel_id = auxlabel_id


def sbreadfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    print("prepare data for ", filename, flush=True)
    f = open(filename, encoding='utf8')
    data = []
    imgs = []
    auxlabels = []
    sentence = []
    label = []
    auxlabel = []
    imgid = ''
    a = 0
    # debug counter for sbreadfile logging
    sbread_debug_counter = 0
    for line in f:
        if line.startswith('IMGID:'):
            # strip whitespace around the id part to avoid leading/trailing spaces
            raw_id = line.strip().split('IMGID:')[1].strip()
            # If IMGID is empty, keep it empty; if it already has an extension, keep as-is,
            # otherwise append '.jpg'. This avoids creating names like 'foo.jpg.jpg'
            if raw_id == '':
                imgid = ''
            else:
                name_no_ext, ext = os.path.splitext(raw_id)
                if ext == '':
                    imgid = raw_id + '.jpg'
                else:
                    imgid = raw_id
            continue

        if line[0] == "\n":
            if len(sentence) > 0:
                # append collected sentence and labels as-is
                data.append((sentence, label))
                imgs.append(imgid)
                auxlabels.append(auxlabel)
                # Debug: print first few parsed samples
                if sbread_debug_counter < 6:
                    print(
                        f"[sbreadfile-main] sample #{len(data)-1} imgid='{imgid}'", flush=True)
                    print(
                        f"[sbreadfile-main] tokens={sentence[:100]}", flush=True)
                    print(
                        f"[sbreadfile-main] labels={label[:100]}", flush=True)
                sentence = []
                label = []
                imgid = ''
                auxlabel = []
            continue
        splits = line.rstrip('\n').split('\t')

        token = splits[0] if len(splits) > 0 else ""
        if token == '' or token.isspace() or token in SPECIAL_TOKENS or token.startswith(URL_PREFIX):
            token = "<unk>"

        sentence.append(token)
        # robust label extraction: if missing, default to 'O'
        if len(splits) >= 2 and splits[-1].strip() != '':
            cur_label = splits[-1].strip()
        else:
            cur_label = 'O'

        if cur_label == 'B-OTHER':
            cur_label = 'B-MISC'
        elif cur_label == 'I-OTHER':
            cur_label = 'I-MISC'
        label.append(cur_label)
        auxlabel.append(cur_label[0] if len(cur_label) > 0 else 'O')

        # Debug: print per-line parsing for first N lines
        if sbread_debug_counter < 50:
            print(
                f"[sbreadfile-main-line] token='{token}' label='{cur_label}' raw='{line[:160]}'", flush=True)
        sbread_debug_counter += 1

    if len(sentence) > 0:
        # append final sample as-is
        data.append((sentence, label))
        imgs.append(imgid)
        auxlabels.append(auxlabel)
        sentence = []
        label = []
        auxlabel = []

    print("The number of samples: " + str(len(data)), flush=True)
    print("The number of images: " + str(len(imgs)), flush=True)
    return data, imgs, auxlabels


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_sbtsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return sbreadfile(input_file)


class MNERProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_sbtsv(
            os.path.join(data_dir, "train.txt"))
        return self._create_examples(data, imgs, auxlabels, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_sbtsv(
            os.path.join(data_dir, "dev.txt"))
        return self._create_examples(data, imgs, auxlabels, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_sbtsv(
            os.path.join(data_dir, "test.txt"))
        return self._create_examples(data, imgs, auxlabels, "test")

    def get_labels(self):
        # Explicit label set as requested
        return [
            "B-ORG",
            "B-MISC",
            "I-PER",
            "I-ORG",
            "B-LOC",
            "I-MISC",
            "I-LOC",
            "O",
            "B-PER",
            "B-NUM",
            "I-NUM",
            "B-DATE",
            "I-DATE",
            "B-OTHER",
            "I-OTHER",
            "X",
            "<s>",
            "</s>",
        ]

        # vlsp2018
        # return ["O","I-ORGANIZATION","B-ORGANIZATION","I-LOCATION","B-MISCELLANEOUS","I-PERSON","B-PERSON","I-MISCELLANEOUS","B-LOCATION","X","<s>","</s>"]

    def get_auxlabels(self):
        return ["O", "B", "I", "X", "<s>", "</s>"]

    def get_start_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['<s>']

    def get_stop_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['</s>']

    def _create_examples(self, lines, imgs, auxlabels, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            img_id = imgs[i]
            label = label
            auxlabel = auxlabels[i]
            examples.append(
                SBInputExample(guid=guid, text_a=text_a, text_b=text_b, img_id=img_id, label=label, auxlabel=auxlabel))
        return examples


def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image


def convert_mm_examples_to_features(examples, label_list, auxlabel_list,
                                    max_seq_length, tokenizer, crop_size, path_img):

    label_map = {label: i for i, label in enumerate(label_list, 1)}
    auxlabel_map = {label: i for i, label in enumerate(auxlabel_list, 1)}

    features = []
    count = 0
    ti_crop_size = 32

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        # args.crop_size, by default it is set to be 224
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    transform_for_ti = transforms.Compose([
        transforms.Resize([ti_crop_size, ti_crop_size]),  # 调整图片到指定的大小
        transforms.ToTensor(),
        transforms.Normalize((0.48, 0.498, 0.531),
                             (0.214, 0.207, 0.207))])

    for (ex_index, example) in enumerate(examples):
        # split on any whitespace to avoid empty tokens when multiple spaces
        textlist = example.text_a.split()
        labellist = example.label
        auxlabellist = example.auxlabel
        tokens = []
        labels = []
        auxlabels = []
        for i, word in enumerate(textlist):
            # If there are fewer labels than words, assign default 'O' instead of crashing
            if i >= len(labellist):
                label_1 = 'O'
                auxlabel_1 = 'O'
            else:
                label_1 = labellist[i]
                auxlabel_1 = auxlabellist[i]

            token = tokenizer.tokenize(word)
            tokens.extend(token)
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    auxlabels.append(auxlabel_1)
                else:
                    labels.append("X")
                    auxlabels.append("X")

        # Debug: print tokenization mapping for first few examples to verify alignment
        if ex_index < 5:
            print(
                f"[tok-main] ex_index={ex_index} original_words={textlist[:30]}", flush=True)
            print(f"[tok-main] mapped_tokens={tokens[:60]}", flush=True)
            print(f"[tok-main] mapped_labels={labels[:60]}", flush=True)

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            auxlabels = auxlabels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        auxlabel_ids = []
        ntokens.append("<s>")
        segment_ids.append(0)
        label_ids.append(label_map["<s>"])
        auxlabel_ids.append(auxlabel_map["<s>"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
            auxlabel_ids.append(auxlabel_map[auxlabels[i]])
        ntokens.append("</s>")
        segment_ids.append(0)
        label_ids.append(label_map["</s>"])
        auxlabel_ids.append(auxlabel_map["</s>"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        # 1 or 49 is for encoding regional image representations
        added_input_mask = [1] * (len(input_ids) + 49)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            added_input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            auxlabel_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(auxlabel_ids) == max_seq_length

        image_name = example.img_id

        # Resolve image file robustly. Support cases where images are all in one
        # folder and filenames may have different extensions or the img_id
        # may or may not include the extension.
        def find_image_file(root_dir, img_name):
            # If img_name already looks like a path with extension, check directly
            candidate = os.path.join(root_dir, img_name)
            # Only accept candidate if it's a file. If it's a directory (or empty), treat as not found
            if os.path.exists(candidate) and os.path.isfile(candidate):
                return candidate

            # Try common extensions. If img_name already has an extension but the exact file
            # is missing, attempt other common extensions (e.g., .png when .jpg missing).
            name_no_ext, ext = os.path.splitext(img_name)
            common_exts = ['.jpg', '.jpeg', '.png', '.bmp']
            if ext != '':
                # image name had an extension but the file wasn't found; try other extensions
                for e in common_exts:
                    if e.lower() == ext.lower():
                        continue
                    candidate = os.path.join(root_dir, name_no_ext + e)
                    if os.path.exists(candidate) and os.path.isfile(candidate):
                        return candidate
            else:
                # no extension provided; try common extensions
                for e in common_exts:
                    candidate = os.path.join(root_dir, name_no_ext + e)
                    if os.path.exists(candidate) and os.path.isfile(candidate):
                        return candidate

            # Fallback: search the directory for a filename that contains the img id
            # (useful when filenames are prefixed/suffixed)
            try:
                for fname in os.listdir(root_dir):
                    if name_no_ext in fname:
                        return os.path.join(root_dir, fname)
            except Exception:
                pass

            # If nothing found, return None
            return None

        image_path = find_image_file(path_img, image_name)

        if image_path is None or not os.path.exists(image_path):
            # log a concise message (do not spam)
            if 'NaN' not in (image_name or ''):
                print(f"Image not found for '{image_name}' in '{path_img}'")
            count += 1
            # Try to find a background.jpg in the image folder first
            candidate_bg = os.path.join(path_img, 'background.jpg')
            if os.path.exists(candidate_bg):
                image_path = candidate_bg
            else:
                # Fallback: try repo's sample_data/ner_image/background.jpg
                repo_root = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), '..', '..'))
                sample_bg = os.path.join(
                    repo_root, 'sample_data', 'ner_image', 'background.jpg')
                if os.path.exists(sample_bg):
                    image_path = sample_bg
                else:
                    # keep candidate_bg (may not exist) so image_process will raise or behave as before
                    image_path = candidate_bg

        # Load image (this may still raise if background.jpg missing)
        image = image_process(image_path, transform)
        image_ti_feat = image_process(image_path, transform_for_ti)

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

            # logger.info(f"img_feat: {image.shape}")
            # logger.info(f"img_ti_feat: {image_ti_feat.shape}")

            logger.info("label: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("auxlabel: %s" % " ".join(
                [str(x) for x in auxlabel_ids]))

        features.append(
            SBInputFeatures(input_ids=input_ids, input_mask=input_mask, added_input_mask=added_input_mask,
                            segment_ids=segment_ids, img_feat=image, img_ti_feat=image_ti_feat, label_id=label_ids, auxlabel_id=auxlabel_ids))

    print('the number of problematic samples: ' + str(count))
    return features


if __name__ == "__main__":
    processor = MNERProcessor()
    label_list = processor.get_labels()
    auxlabel_list = processor.get_auxlabels()
    # label 0 corresponds to padding, label in label_list starts from 1
    num_labels = len(label_list) + 1

    start_label_id = processor.get_start_label_id()
    stop_label_id = processor.get_stop_label_id()

    data_dir = r'/home/vms/bags/vlsp_all/origin+image/VLSP2018'
    eval_examples = processor.get_test_examples(data_dir)
    print(eval_examples[0].img_id)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    eval_features = convert_mm_examples_to_features(
        eval_examples, label_list, auxlabel_list, 256, tokenizer, 224, data_dir+'/ner_image')

    print(len(eval_features))
