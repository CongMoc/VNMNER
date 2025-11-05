from PIL import Image
from torchvision import transforms
import torch
import logging
import os
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
    print("prepare data for ", filename)
    f = open(filename, encoding='utf8')
    data = []
    imgs = []
    auxlabels = []
    sentence = []
    label = []
    auxlabel = []
    imgid = ''
    a = 0
    for line in f:
        # Handle both IMGID: and IMID: (common typo in data)
        if line.startswith('IMGID:') or line.startswith('IMID:'):
            # Extract ID from either IMGID: or IMID:
            if line.startswith('IMGID:'):
                raw_id = line.strip().split('IMGID:')[1].strip()
            else:
                raw_id = line.strip().split('IMID:')[1].strip()
            
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
                data.append((sentence, label))
                imgs.append(imgid)
                auxlabels.append(auxlabel)
                sentence = []
                label = []
                imgid = ''
                auxlabel = []
            continue
        splits = line.split('\t')

        if splits[0] == '' or splits[0].isspace() or splits[0] in SPECIAL_TOKENS or splits[0].startswith(URL_PREFIX):
            splits[0] = "[UNK]"

        sentence.append(splits[0])
        cur_label = splits[-1][:-1]
        if cur_label == 'B-OTHER':
            cur_label = 'B-MISC'
        elif cur_label == 'I-OTHER':
            cur_label = 'I-MISC'
        label.append(cur_label)
        auxlabel.append(cur_label[0])

    if len(sentence) > 0:
        data.append((sentence, label))
        imgs.append(imgid)
        auxlabels.append(auxlabel)
        sentence = []
        label = []
        auxlabel = []

    print("The number of samples: " + str(len(data)))
    print("The number of images: " + str(len(imgs)))
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
        # Custom Vietnamese MNER dataset (sonba)
        # Has 7 entity types: DATE, LOC, MISC, NUM, ORG, OTHER, PER
        return [
            "O",
            "B-DATE",
            "I-DATE",
            "B-LOC",
            "I-LOC",
            "B-MISC",
            "I-MISC",
            "B-NUM",
            "I-NUM",
            "B-ORG",
            "I-ORG",
            "B-OTHER",
            "I-OTHER",
            "B-PER",
            "I-PER",
            "X",
            "[CLS]",
            "[SEP]"]

        # vlsp2018
        # return ["O","I-ORGANIZATION","B-ORGANIZATION","I-LOCATION","B-MISCELLANEOUS","I-PERSON","B-PERSON","I-MISCELLANEOUS","B-LOCATION","X","[CLS]","[SEP]"]

    def get_auxlabels(self):
        return ["O", "B", "I", "X", "[CLS]", "[SEP]"]

    def get_start_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['[CLS]']

    def get_stop_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['[SEP]']

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
        textlist = example.text_a.split(' ')
        labellist = example.label
        auxlabellist = example.auxlabel
        
        # Debug: check if lengths match
        if len(textlist) != len(labellist):
            print(f"Warning: Example {ex_index} has mismatched lengths - text: {len(textlist)}, labels: {len(labellist)}")
            print(f"  Text: {textlist[:10]}")
            print(f"  Labels: {labellist[:10]}")
            # Skip this example or truncate to match
            min_len = min(len(textlist), len(labellist))
            textlist = textlist[:min_len]
            labellist = labellist[:min_len]
            auxlabellist = auxlabellist[:min_len]
        
        tokens = []
        labels = []
        auxlabels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            auxlabel_1 = auxlabellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    auxlabels.append(auxlabel_1)
                else:
                    labels.append("X")
                    auxlabels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            auxlabels = auxlabels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        auxlabel_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        auxlabel_ids.append(auxlabel_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
            auxlabel_ids.append(auxlabel_map[auxlabels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        auxlabel_ids.append(auxlabel_map["[SEP]"])
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

        def find_image_file(root_dir, img_name):
            candidate = os.path.join(root_dir, img_name)
            if os.path.exists(candidate) and os.path.isfile(candidate):
                return candidate
            name_no_ext, ext = os.path.splitext(img_name)
            common_exts = ['.jpg', '.jpeg', '.png', '.bmp']
            if ext != '':
                for e in common_exts:
                    if e.lower() == ext.lower():
                        continue
                    candidate = os.path.join(root_dir, name_no_ext + e)
                    if os.path.exists(candidate) and os.path.isfile(candidate):
                        return candidate
            else:
                for e in common_exts:
                    candidate = os.path.join(root_dir, name_no_ext + e)
                    if os.path.exists(candidate) and os.path.isfile(candidate):
                        return candidate
            try:
                for fname in os.listdir(root_dir):
                    if name_no_ext in fname and os.path.isfile(os.path.join(root_dir, fname)):
                        return os.path.join(root_dir, fname)
            except Exception:
                pass
            return None

        image_path = find_image_file(path_img, image_name)

        if image_path is None or not os.path.exists(image_path):
            if 'NaN' not in (image_name or ''):
                print(f"Image not found for '{image_name}' in '{path_img}'")
            count += 1
            candidate_bg = os.path.join(path_img, 'background.jpg')
            if os.path.exists(candidate_bg) and os.path.isfile(candidate_bg):
                image_path = candidate_bg
            else:
                repo_root = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), '..', '..'))
                sample_bg = os.path.join(
                    repo_root, 'sample_data', 'ner_image', 'background.jpg')
                if os.path.exists(sample_bg) and os.path.isfile(sample_bg):
                    image_path = sample_bg
                else:
                    image_path = candidate_bg

        try:
            image = image_process(image_path, transform)
            image_ti_feat = image_process(image_path, transform_for_ti)
        except Exception:
            count += 1
            image_path_fail = os.path.join(path_img, 'background.jpg')
            image = image_process(image_path_fail, transform)
            image_ti_feat = image_process(image_path_fail, transform_for_ti)

        else:
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

                logger.info("label: %s" % " ".join(
                    [str(x) for x in label_ids]))
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

    data_dir = r'sample_data'
    train_examples = processor.get_train_examples(data_dir)
    print(train_examples[0].img_id)
