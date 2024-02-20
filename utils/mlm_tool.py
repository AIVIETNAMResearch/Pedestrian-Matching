import torch
import math
from random import randint, shuffle, sample
from random import random as rand
import spacy

class TextMaskingGenerator:
    def __init__(self, tokenizer, mask_prob, mask_max, skipgram_prb=0.2, skipgram_size=3, mask_whole_word=True,
                 use_roberta=False):
        self.id2token = {i: w for w, i in tokenizer.get_vocab().items()}
        self.use_roberta = use_roberta
        for i in range(len(self.id2token)):
            assert i in self.id2token.keys()  # check
        self.cls_token_id = tokenizer.cls_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.mask_max = mask_max
        self.mask_prob = mask_prob
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word

        # print("len(tokenizer.id2token): ", len(self.id2token), "  ----  cls_token_id: ", self.cls_token_id,
        #       "  ----  mask_token_id: ", self.mask_token_id, flush=True)

    def get_random_word(self):
        i = randint(0, len(self.id2token) - 1)
        return i  # self.id2token[i]

    def __call__(self, text_ids):  # tokens: [CLS] + ...
        n_pred = min(self.mask_max, max(1, int(round(len(text_ids) * self.mask_prob))))

        # candidate positions of masked tokens
        assert text_ids[0] == self.cls_token_id
        special_pos = set([0])  # will not be masked
        cand_pos = list(range(1, len(text_ids)))

        shuffle(cand_pos)
        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end

                if self.use_roberta:
                    while (new_st > 1) and (self.id2token[text_ids[new_st].item()][0] != 'Ġ'):
                        new_st -= 1
                    while (new_end < len(text_ids)) and (self.id2token[text_ids[new_end].item()][0] != 'Ġ'):
                        new_end += 1
                else:
                    # bert, WordPiece
                    while (new_st >= 0) and self.id2token[text_ids[new_st].item()].startswith('##'):
                        new_st -= 1
                    while (new_end < len(text_ids)) and self.id2token[text_ids[new_end].item()].startswith('##'):
                        new_end += 1

                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        n_real_pred = len(masked_pos)
        if n_real_pred > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                text_ids[pos] = self.mask_token_id
            elif rand() < 0.5:  # 10%
                text_ids[pos] = self.get_random_word()

        return text_ids, masked_pos
    
class NounMaskingGenerator:
    def __init__(self, tokenizer, mask_prob, mask_max, skipgram_prb=0.2, skipgram_size=3, mask_whole_word=True,
                 use_roberta=False):
        self.tokenizer = tokenizer
        self.mask_max = mask_max
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.mask_token_id
        self.nlp = spacy.load("en_core_web_sm")  # Load the spaCy model
    
    def mask_nouns(self, text):
        # Tokenize the text with offset mapping
        encoded = self.tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=True)
        text_ids = encoded.input_ids
        offset_mapping = encoded.offset_mapping

        # Process the text with spaCy to find noun positions
        doc = self.nlp(text)
        noun_positions = []
        for token in doc:
            if token.pos_ == "NOUN":
                # Find the token index for each character position in the noun
                start, end = token.idx, token.idx + len(token.text)
                for idx, (offset_start, offset_end) in enumerate(offset_mapping):
                    if start >= offset_start and end <= offset_end:
                        noun_positions.append(idx)
                        break  # Assume one token per noun for simplicity

        # If there are more nouns than max_mask, randomly select max_mask nouns to mask
        if len(noun_positions) > self.mask_max:
            noun_positions = sample(noun_positions, self.mask_max)

        # Mask the selected nouns and keep track of the original IDs
        masked_positions = []
        masked_ids = []
        for idx in noun_positions:
            # Ensure not to mask special tokens like [CLS] and [SEP]
            if idx != 0 and idx < len(text_ids) - 1:
                masked_positions.append(idx)
                masked_ids.append(text_ids[idx])  # Store the original token ID
                text_ids[idx] = self.mask_token_id  # Replace with mask token ID

        return text_ids, masked_positions, masked_ids

def mlm(text, text_input, tokenizer, device, mask_generator, config):
    if type(mask_generator) == TextMaskingGenerator:
        text_masked = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                    return_tensors="pt").to(device)
        text_ids_masked = text_masked.input_ids
        masked_pos = torch.empty((text_ids_masked.shape[0], config['max_masks']), dtype=torch.int64, device=device)
        masked_ids = torch.empty((text_ids_masked.shape[0], config['max_masks']), dtype=torch.long, device=device)
        for index, text_id in enumerate(text_ids_masked):
            text_ids_masked_, masked_pos_ = mask_generator(text_id)
            masked_ids_ = [text_input.input_ids[index][p].item() for p in masked_pos_]
            n_pad = config['max_masks'] - len(masked_ids_)
            masked_pos_ = masked_pos_ + [0] * n_pad
            masked_pos_ = torch.tensor(masked_pos_, dtype=torch.int64).to(device)
            masked_ids_ = masked_ids_ + [-100] * n_pad
            masked_ids_ = torch.tensor(masked_ids_, dtype=torch.long).to(device)
            masked_pos[index] = masked_pos_
            masked_ids[index] = masked_ids_
        return text_ids_masked, masked_pos, masked_ids
    
    elif type(mask_generator) == NounMaskingGenerator:
        text_ids_masked = torch.empty((len(text), config['max_tokens']), dtype=torch.long, device=device).to(device)
        masked_pos = torch.empty((len(text), config['max_masks']), dtype=torch.int64, device=device).to(device)
        masked_ids = torch.empty((len(text), config['max_masks']), dtype=torch.long, device=device).to(device)
        # Check the number of vocab in a tokenizer 
        for index, single_text in enumerate(text):
            # Generate masked text using the mask generator
            masked_text_ids, masked_pos_, masked_ids_ = mask_generator.mask_nouns(single_text)

            # Pad or truncate masked_text_ids to max_tokens length
            if len(masked_text_ids) > config['max_tokens']:
                masked_text_ids = masked_text_ids[:config['max_tokens']]
            else:
                masked_text_ids += [tokenizer.pad_token_id] * (config['max_tokens'] - len(masked_text_ids))

            # Convert to tensor and assign to text_ids_masked
            text_ids_masked[index] = torch.tensor(masked_text_ids, dtype=torch.long).to(device)

            # Prepare masked_pos and masked_ids for the current text
            n_pad = config['max_masks'] - len(masked_pos_)
            masked_pos_ += [0] * n_pad  # Pad masked_indices if necessary
            masked_pos[index] = torch.tensor(masked_pos_, dtype=torch.int64).to(device)

            # Retrieve the original IDs of the masked tokens and pad if necessary
            original_ids = masked_ids_ + [0] * n_pad
            masked_ids[index] = torch.tensor(original_ids, dtype=torch.long).to(device)
        
        return text_ids_masked, masked_pos, masked_ids
    else:
        raise ValueError("mask_generator must be an instance of TextMaskingGenerator or NounMaskingGenerator")