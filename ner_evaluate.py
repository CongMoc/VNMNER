import codecs
import numpy as np 

def get_chunks(seq, tags):
    """
    Robust chunk extraction supporting two forms of `tags`:
      - tag2idx: {'O': 4, 'B-PER': 1, ...}
      - idx2tag: {0: 'O', 1: 'B-PER', ...}

    Returns list of (chunk_type, chunk_start, chunk_end)
    """
    # Detect whether tags is tag->idx or idx->tag
    is_tag2idx = False
    is_idx2tag = False
    try:
        # simple heuristic: if any key is a string and value is int => tag2idx
        if any(isinstance(k, str) for k in tags.keys()) and any(isinstance(v, int) for v in tags.values()):
            is_tag2idx = True
        if any(isinstance(k, int) for k in tags.keys()) and any(isinstance(v, str) for v in tags.values()):
            is_idx2tag = True
    except Exception:
        pass

    if is_tag2idx:
        tag2idx = tags
        idx_to_tag = {idx: tag for tag, idx in tag2idx.items()}
        # find default index for 'O'
        if 'O' in tag2idx:
            default = tag2idx['O']
        else:
            # try common alternatives
            default = tag2idx.get('0') or tag2idx.get('o') if isinstance(tag2idx.get('0'), int) else None
            if default is None:
                # try to find tag whose type part is 'O'
                for tag, idx in tag2idx.items():
                    if isinstance(tag, str) and tag.split('-')[-1].upper() == 'O':
                        default = idx
                        break
            if default is None:
                # fallback to 0
                default = 0
    elif is_idx2tag:
        idx_to_tag = tags
        # find index for 'O' in values
        default = None
        for idx, tag in idx_to_tag.items():
            if isinstance(tag, str) and tag == 'O':
                default = idx
                break
        if default is None:
            for idx, tag in idx_to_tag.items():
                if isinstance(tag, str) and tag.split('-')[-1].upper() == 'O':
                    default = idx
                    break
        if default is None:
            default = 0
    else:
        # Unknown mapping shape; try to treat as tag->idx by default
        try:
            idx_to_tag = {idx: tag for tag, idx in tags.items()}
            default = tags.get('O', tags.get('0', 0))
        except Exception:
            idx_to_tag = {}
            default = 0

    chunks = []
    chunk_type, chunk_start = None, None

    for i, tok in enumerate(seq):
        # Skip tokens not present in idx_to_tag (padding or unknown)
        if tok not in idx_to_tag:
            continue

        # End of a chunk: token equals default 'O'
        if tok == default and chunk_type is not None:
            chunks.append((chunk_type, chunk_start, i))
            chunk_type, chunk_start = None, None

        # Process non-default tokens
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunks.append((chunk_type, chunk_start, i))
                chunk_type, chunk_start = tok_chunk_type, i

    # Add final chunk if any
    if chunk_type is not None:
        chunks.append((chunk_type, chunk_start, len(seq)))

    return chunks

def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, e.g. 4
        idx_to_tag: dictionary mapping id->tag string
    Returns:
        tuple: (class, type) e.g. ("B", "PER")
    """
    tag_name = idx_to_tag.get(tok, 'O')
    parts = tag_name.split('-') if isinstance(tag_name, str) else ['O']
    tag_class = parts[0] if len(parts) >= 1 else 'O'
    tag_type = parts[-1] if len(parts) >= 2 else parts[0]
    return tag_class, tag_type

# def run_evaluate(self, sess, test, tags):
def evaluate(labels_pred, labels,words,tags):

	"""
	words,pred, right: is a sequence, is label index or word index.
	Evaluates performance on test set
	Args:
		sess: tensorflow session
		test: dataset that yields tuple of sentences, tags
		tags: {tag: index} dictionary
	Returns:
		accuracy
		f1 score
		...
	"""

	#file_write = open('./test_results.txt','w')


	index = 0 
	sents_length = []

	accs = []
	correct_preds, total_correct, total_preds = 0., 0., 0.


	for lab, lab_pred, word_sent in zip(labels, labels_pred, words):
		word_st = word_sent
		lab = lab
		lab_pred = lab_pred
		accs += [a==b for (a, b) in zip(lab, lab_pred)]
		lab_chunks = set(get_chunks(lab, tags))
		lab_pred_chunks = set(get_chunks(lab_pred, tags))
		correct_preds += len(lab_chunks & lab_pred_chunks)
		total_preds += len(lab_pred_chunks)
		total_correct += len(lab_chunks)

		#for i in range(len(word_st)):
				#file_write.write('%s\t%s\t%s\n'%(word_st[i],lab[i],lab_pred[i]))
		#file_write.write('\n')

	p = correct_preds / total_preds if correct_preds > 0 else 0
	r = correct_preds / total_correct if correct_preds > 0 else 0
	f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
	acc = np.mean(accs)

	#file_write.close()
	return acc, f1,p,r

def evaluate_each_class(labels_pred, labels,words,tags, class_type):
		#class_type:PER or LOC or ORG
		index = 0

		accs = []
		correct_preds, total_correct, total_preds = 0., 0., 0.
		correct_preds_cla_type, total_preds_cla_type, total_correct_cla_type = 0., 0., 0.

		for lab, lab_pred, word_sent in zip(labels, labels_pred, words):
				lab_pre_class_type = []
				lab_class_type=[]

				word_st = word_sent
				lab = lab
				lab_pred = lab_pred
				lab_chunks = get_chunks(lab, tags)
				lab_pred_chunks = get_chunks(lab_pred, tags)
				for i in range(len(lab_pred_chunks)):
						if lab_pred_chunks[i][0] ==class_type:
								lab_pre_class_type.append(lab_pred_chunks[i])
				lab_pre_class_type_c = set(lab_pre_class_type)

				for i in range(len(lab_chunks)):
						if lab_chunks[i][0] ==class_type:
								lab_class_type.append(lab_chunks[i])
				lab_class_type_c = set(lab_class_type)
				
				lab_chunksss = set(lab_chunks) 
				correct_preds_cla_type +=len(lab_pre_class_type_c & lab_chunksss)
				total_preds_cla_type +=len(lab_pre_class_type_c)
				total_correct_cla_type += len(lab_class_type_c)

		p = correct_preds_cla_type / total_preds_cla_type if correct_preds_cla_type > 0 else 0
		r = correct_preds_cla_type / total_correct_cla_type if correct_preds_cla_type > 0 else 0
		f1 = 2 * p * r / (p + r) if correct_preds_cla_type > 0 else 0

		return f1,p,r


if __name__ == '__main__':
		max_sent=10
		tags = {'0':0,
		'B-PER':1, 'I-PER':2,
		'B-LOC':3, 'I-LOC':4,
		'B-ORG':5, 'I-ORG':6,
		'B-OTHER':7, 'I-OTHER':8,
		'O':9}
		labels_pred=[
								[9,9,9,1,3,1,2,2,0,0],
								[9,9,9,1,3,1,2,0,0,0]
		]
		labels = [
						[9,9,9,9,3,1,2,2,0,0],
						[9,9,9,9,3,1,2,2,0,0]
						]
		words = [
						[0,0,0,0,0,3,6,8,5,7],
						[0,0,0,4,5,6,7,9,1,7]
						]
		id_to_vocb = {0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h',8:'i',9:'j'}
		# new_words = []
		# for i in range(len(words)):
		# 	sent = []
		# 	for j in range(len(words[i])):
		# 		sent.append(id_to_vocb[words[i][j]])
		# 	new_words.append(sent)
		# class_type = 'PER'
		# acc, f1,p,r = evaluate(labels_pred, labels,new_words,tags)
		# print(p,r,f1)
		# f1,p,r = evaluate_each_class(labels_pred, labels,new_words,tags, class_type)
		# print(p,r,f1)

		acc, f1, p, r = evaluate(labels_pred, labels, words, tags)
		print(acc)