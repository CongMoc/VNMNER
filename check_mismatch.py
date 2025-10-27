# check_mismatch.py
# Scans sample_data train/dev/test files and reports samples where number of words != number of labels
from pathlib import Path
SPECIAL_TOKENS = ['\ufe0f', '\u200d', '\u200b', '\x92']
URL_PREFIX = 'http'

def check_file(p):
    p = Path(p)
    if not p.exists():
        print(f"File not found: {p}")
        return
    sentence=[]
    label=[]
    imgid=''
    idx = 0
    with p.open(encoding='utf8') as f:
        for line in f:
            if line.startswith('IMGID:'):
                imgid = line.strip().split('IMGID:')[1]
                continue
            if line.strip() == '':
                if sentence:
                    if len(sentence) != len(label):
                        print(f"Mismatch in {p.name} sample #{idx} imgid={imgid}: words={len(sentence)}, labels={len(label)}")
                        print('  words sample:', sentence[:20])
                        print('  labels sample:', label[:20])
                    idx += 1
                    sentence=[]; label=[]; imgid=''
                continue
            splits = line.split('\t')
            token = splits[0]
            if token == '' or token.isspace() or token in SPECIAL_TOKENS or token.startswith(URL_PREFIX):
                token = '<unk>'
            sentence.append(token)
            cur_label = splits[-1].strip()
            # if cur_label may be empty, keep as empty
            label.append(cur_label)
    if sentence:
        if len(sentence) != len(label):
            print(f"FINAL mismatch in {p.name} imgid={imgid}: words={len(sentence)}, labels={len(label)}")
            print('  words sample:', sentence[:20])
            print('  labels sample:', label[:20])

if __name__ == '__main__':
    base = Path('sample_data')
    for name in ['train.txt','dev.txt','test.txt']:
        check_file(base / name)
