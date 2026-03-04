import json
import re
from collections import Counter

#Added import
from typing import List, Optional
import torch


class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        elif self.dataset_name == 'wsi_report':
            self.clean_report = self.clean_report_pathology
        else:
            self.clean_report = self.clean_report_mimic_cxr
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report
    
    def clean_report_pathology(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
    
class ModernTokenizer(object):
    """A wrapper class that can use different modern tokenizers while maintaining compatibility
    with your existing codebase."""
    
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset_name
        
        # Option 1: Medical-specific tokenizer (BioClinicalBERT)
        # self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        
        # Option 2: General purpose tokenizer (RoBERTa)
        # self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
        # Option 3: Domain-adapted tokenizer (PubMedBERT)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        
        # Add special tokens specific to medical reports if needed
        special_tokens = {
            "additional_special_tokens": [
                "<impression>", "</impression>",
                "<findings>", "</findings>",
                "<comparison>", "</comparison>",
                "<indication>", "</indication>"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Load your existing vocabulary for compatibility
        self.ann = json.loads(open(args.ann_path, 'r').read())
        
        # Map special tokens
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token
        
        # Create reverse mapping
        self.token2idx = self.tokenizer.get_vocab()
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        
    def __call__(self, report: str) -> List[int]:
        """Convert report text to token ids."""
        # Use the tokenizer's built-in encoding
        encoding = self.tokenizer.encode(
            report,
            add_special_tokens=True,
            max_length=self.args.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding[0].tolist()  # Convert tensor to list
    
    def decode(self, ids: List[int]) -> str:
        """Convert token ids back to text."""
        # Use the tokenizer's built-in decoding
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return self.post_process_medical_text(text)

    def post_process_medical_text(self, text: str) -> str:
        """Apply domain-specific post-processing to medical reports."""
        # Capitalize anatomical terms
        anatomical_terms = {'chest', 'heart', 'lungs', 'pleural', 'mediastinum'}
        for term in anatomical_terms:
            text = text.replace(f' {term} ', f' {term.capitalize()} ')
        
        # Fix common medical abbreviations
        abbreviations = {
            'pa': 'PA',
            'ap': 'AP',
            'lat': 'LAT',
            'iv': 'IV',
            'ct': 'CT'
        }
        for abbr, replacement in abbreviations.items():
            text = text.replace(f' {abbr} ', f' {replacement} ')
        
        # Fix measurements spacing - using a safer regex pattern
        text = re.sub(r'(\d+)(mm|cm|inches)', r'\1 \2', text)
        
        # Proper sentence capitalization
        sentences = text.split('. ')
        sentences = [s.capitalize() for s in sentences if s]
        text = '. '.join(sentences)
        
        return text.strip()
    
    def decode_batch(self, ids_batch: List[List[int]]) -> List[str]:
        """Decode a batch of token ids to texts."""
        return [self.decode(ids) for ids in ids_batch]
    
    def get_vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.token2idx)
    
    def get_token_by_id(self, id: int) -> str:
        """Get token string from token id."""
        return self.idx2token.get(id, self.unk_token)
    
    def get_id_by_token(self, token: str) -> int:
        """Get token id from token string."""
        return self.token2idx.get(token, self.token2idx[self.unk_token])
        
    def batch_encode(self, reports: List[str], 
                    max_length: Optional[int] = None) -> torch.Tensor:
        """Encode a batch of reports to token ids."""
        if max_length is None:
            max_length = self.args.max_seq_length
            
        encodings = self.tokenizer.batch_encode_plus(
            reports,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encodings['input_ids']
    
class MedicalReportTokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        
        # Special tokens
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        
        # My additional vocal
        

        final_additions = {
            # (could be resulting in unk? Test it)
            
            # Essential punctuation
            '(', ')', '+', ':', ',', ';',
            
            # Gleason score patterns (both formats)
            'gleason', "gleason's", 'score',
            '(3+3)', '(3+4)', '(4+3)', '(4+4)', '(4+5)', '(5+4)', '(5+5)',
            '3+3', '3+4', '4+3', '4+4', '4+5', '5+4', '5+5',
            
            # Individual numbers (critical!)
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
            
            # Grade patterns
            'grade', 'group',
            
            # Tumor grading terms
            'tubule', 'formation', 'nuclear', 'mitoses', 'tumor', 'volume',
            
            # Specific percentages from your data
            '1%', '5%)', '7%', '33%', '35%', '55%', '65%', '66%', '75%',
            
            # Medical abbreviations from your data
            'g1', 'nos', 'spp', 'micro-invasive',
            
            # === HIGH-VALUE ADDITIONS FROM GENERAL MEDICAL KNOWLEDGE ===
            # (These are likely to appear and cause issues)
            
            # Common grade group patterns
            'group1', 'group2', 'group3', 'group4', 'group5',
            
            # Essential medical terms
            'ductal', 'invasive', 'situ', 'solid', 'comedo', 'type',
            'necrosis', 'present', 'acinar', 'biopsy',
            'adenocarcinoma', 'carcinoma', 'moderately', 'differentiated',
            
            # Common medical connectors
            'with', 'without', 'of', 'in', 'no', 'absent',
            
            # Additional percentages (common in medical reports)
            '10%', '20%', '30%', '40%', '45%', '50%', '60%', '70%', '80%', '90%', '100%',
            
            # Roman numerals (for grading)
            'i', 'ii', 'iii', 'I', 'II', 'III',
            
            # Measurement units
            'mm', 'cm', 'ml', 'cc', 'µm', '%',
            
            # Essential descriptors
            'chronic', 'well', 'poorly', 'evidence', 'specimen',
        }
        
        # Add medical domain vocabulary
        self.domain_vocab = set([
            # Common anatomical terms
            'granuloma', 'parametria', 'lymph', 'nodes', 'cervix', 'uterus',
            'ovary', 'ovaries', 'fallopian', 'tubes', 'endometrium', 'myometrium',
            'serosa', 'parametrium', 'vagina', 'vulva',
            
            # Common pathological terms
            'carcinoma', 'adenocarcinoma', 'sarcoma', 'lymphoma', 'metastasis',
            'neoplasm', 'tumor', 'lesion', 'mass', 'cyst', 'polyp', 'nodule', 'scars', 'adult',
            'dome-shaped', 'flap', 'prophylactic',
            
            # Descriptive terms
            'focal', 'diffuse', 'acute', 'chronic', 'mild', 'moderate', 'severe',
            'benign', 'malignant', 'invasive', 'metastatic', 'astrocytomas', 'anaplastic', 'cataractous',
            
            "mm.", "cm.", "%", "mm,", "cm,", "%,",
            
            # Add measurement combinations
            *[f"{i}mm" for i in range(1, 101)],  # 1mm to 100mm
            *[f"{i}cm" for i in range(1, 51)],   # 1cm to 50cm
            *[f"{i}%" for i in range(1, 100)],   # 1cm to 50cm
            *[f"{i}.{j}" for i in range(10) for j in range(10)],  # Common decimal numbers
            *[f"{i}.{j}cm" for i in range(10) for j in range(10)],
            *[f"{i}.{j}mm" for i in range(10) for j in range(10)],
            *[f"{i}.{j}%" for i in range(10) for j in range(10)],
            *[f"{i}th" for i in range(1, 101)],
        ]) | final_additions 
        
        # Map dataset names to cleaning functions
        self.clean_report_map = {
            'iu_xray': self.clean_report_iu_xray,
            'wsi_report': self.clean_report_pathology,
            'mimic_cxr': self.clean_report_mimic_cxr
        }
        self.clean_report = self.clean_report_map.get(self.dataset_name, self.clean_report_mimic_cxr)
        
        self.preserve_case = (self.dataset_name == 'wsi_report')
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens = []
        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            total_tokens.extend(tokens)

        counter = Counter(total_tokens)
        vocab_tokens = set(k for k, v in counter.items() if v >= self.threshold)
        vocab_tokens.update(self.domain_vocab)
        vocab_tokens = sorted(list(vocab_tokens))
        
        special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        vocab = special_tokens + vocab_tokens
        
        token2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2token = {idx: token for idx, token in enumerate(vocab)}
        return token2idx, idx2token

    def normalize_whitespace(self, text):
        if self.dataset_name == 'wsi_report':
            text = re.sub(r'\n\s*\n', '\n', text)
            text = re.sub(r' +', ' ', text)
        else:
            text = ' '.join(text.split())
            text = text.replace('\n', ' ')
        return text

    def normalize_punctuation(self, text):
        # Replace multiple periods with single period, but preserve decimal points
        text = re.sub(r'\.(?![\d])', '. ', text)  # Add space after period unless followed by digit
        text = re.sub(r'\.{2,}', '.', text)       # Replace multiple periods
        
        if self.dataset_name == 'wsi_report':
            # Preserve commas in numbers (e.g., 1,000)
            text = re.sub(r'([^0-9]),([^0-9])', r'\1, \2', text)  # Add spaces around non-numeric commas
            text = re.sub(r'([.,;:])\1+', r'\1', text)
        else:
            text = re.sub(r'\d+\.\s*', '', text)
            text = re.sub(r'\s*\.\s*', '. ', text)
        
        return text.strip()

    def clean_sentence(self, sentence):
        if self.dataset_name == 'wsi_report':
            # Preserve numbers, decimal points, and measurements
            # Allow numbers, letters, basic punctuation, and measurement units
            sentence = re.sub(r'[^a-zA-Z0-9\s.,;:()\-\.]', ' ', sentence)
            
            # Ensure measurements stay together (no space between number and unit)
            sentence = re.sub(r'(\d+)\s+(mm|cm|ml|cc|µm)', r'\1\2', sentence)
            
            # Normalize spaces
            sentence = re.sub(r'\s+', ' ', sentence)
            cleaned = sentence.strip()
        else:
            sentence = re.sub(r'[^a-zA-Z0-9\s.,;:]', ' ', sentence)
            cleaned = sentence.lower().strip()
            cleaned = re.sub(r'[.,;:]', '', cleaned)
        
        return cleaned

    def tokenize_with_numbers(self, text):
        """Special tokenization that preserves numbers and measurements."""
        # Split on spaces but preserve special patterns
        tokens = []
        words = text.split()
        
        for word in words:
            # Check if word contains a number pattern
            if re.search(r'\d', word):
                # Keep measurements together (e.g., "7mm", "4.5cm")
                if re.match(r'^\d+\.?\d*(?:mm|cm|ml|cc|µm)?$', word):
                    tokens.append(word)
                else:
                    # Split other numeric patterns
                    numeric_parts = re.findall(r'(\d+\.?\d*)|([^0-9]+)', word)
                    tokens.extend(part for parts in numeric_parts for part in parts if part)
            else:
                tokens.append(word)
        
        return tokens
    #Original
    def __call__(self, report):
        """Convert a report to token ids."""
        cleaned_report = self.clean_report(report)
        
        if self.dataset_name == 'wsi_report':
            tokens = self.tokenize_with_numbers(cleaned_report)
        else:
            tokens = cleaned_report.split()
        
        tokens = [self.bos_token] + tokens + [self.eos_token]
        return [self.get_id_by_token(token) for token in tokens]
    
    

    '''def split_punctuation(text):
        # Put spaces around punctuation so they become separate tokens
        return re.sub(r'([.,;:])', r' \1 ', text)

    def __call__(self, report):
        cleaned_report = self.clean_report(report)
        cleaned_report = split_punctuation(cleaned_report)  # NEW
        if self.dataset_name == 'wsi_report':
            tokens = self.tokenize_with_numbers(cleaned_report)
        else:
            tokens = cleaned_report.split()
        tokens = [self.bos_token] + tokens + [self.eos_token]
        return [self.get_id_by_token(token) for token in tokens]'''


    def clean_report_pathology(self, report):
        """Special handling for pathology reports."""
        # Normalize underscores and whitespace
        report = re.sub(r'_+', ' ', report)
        report = self.normalize_whitespace(report)
        report = self.normalize_punctuation(report)
        
        # Split on periods while preserving them
        sentences = re.split(r'(?<=\.)\s+', report)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Clean each sentence
        cleaned_sentences = [self.clean_sentence(sent) for sent in sentences]
        cleaned_sentences = [sent for sent in cleaned_sentences if sent]
        
        # Join with proper punctuation
        if cleaned_sentences:
            # Add period if not present
            result = '. '.join(s.rstrip('.') for s in cleaned_sentences) + '.'
            # Clean up any double periods
            result = re.sub(r'\.+', '.', result)
            return result
        return ''

    def clean_report_iu_xray(self, report):
        # Normalize whitespace and punctuation
        report = self.normalize_whitespace(report)
        report = self.normalize_punctuation(report)
        
        # Split into sentences and clean each one
        sentences = [s.strip() for s in report.split('.') if s.strip()]
        cleaned_sentences = [self.clean_sentence(sent) for sent in sentences]
        cleaned_sentences = [sent for sent in cleaned_sentences if sent]
        
        # Join sentences with proper spacing
        return ' . '.join(cleaned_sentences) + ' .'

    def clean_report_mimic_cxr(self, report):
        # Additional preprocessing for MIMIC-CXR
        report = re.sub(r'_+', ' ', report)  # Replace underscores with space
        report = self.normalize_whitespace(report)
        report = self.normalize_punctuation(report)
        
        sentences = [s.strip() for s in report.split('.') if s.strip()]
        cleaned_sentences = [self.clean_sentence(sent) for sent in sentences]
        cleaned_sentences = [sent for sent in cleaned_sentences if sent]
        
        return ' . '.join(cleaned_sentences) + ' .'

    def get_token_by_id(self, id):
        return self.idx2token.get(id, self.unk_token)

    #Original function
    def get_id_by_token(self, token):
        # For non-WSI reports, convert to lowercase for token lookup
        if not self.preserve_case:
            token = token.lower()
        return self.token2idx.get(token, self.token2idx[self.unk_token])
    
    #My function
    '''def get_id_by_token(self, token):
        # For non-WSI reports, convert to lowercase for token lookup
        if not self.preserve_case:
            token = token.lower()
        
        # Try exact match first
        if token in self.token2idx:
            return self.token2idx[token]
        
        # If exact match fails, try without trailing punctuation
        clean_token = token.rstrip('.,;:!?')
        if not self.preserve_case:
            clean_token = clean_token.lower()
        
        if clean_token in self.token2idx:
            return self.token2idx[clean_token]
        
        # Fall back to unknown token
        return self.token2idx[self.unk_token]'''

    def get_vocab_size(self):
        return len(self.token2idx)

    #Commented original method
    def decode(self, ids):
        """Convert token ids back to text."""
        tokens = []
        for idx in ids:
            if idx == self.token2idx[self.eos_token]:
                break
            if idx != self.token2idx[self.pad_token]:
                token = self.idx2token[idx]
                if token != self.bos_token:  # Skip BOS token in output
                    tokens.append(token)
        
        # Join tokens with proper spacing
        if self.dataset_name == 'wsi_report':
            # For WSI reports, handle punctuation specially
            text = ''
            for i, token in enumerate(tokens):
                if token in '.,;:':
                    text = text.rstrip() + token + ' '
                else:
                    text += token + ' '
            return text.strip()
        else:
            return ' '.join(tokens)
    
    #My method
    '''def decode(self, ids):
        tokens = []
        for idx in ids:
            if idx == self.token2idx[self.eos_token]:
                break
            if idx != self.token2idx[self.pad_token]:
                token = self.idx2token[idx]
                tokens.append(token)

        # Remove BOS only if it's at the beginning
        if tokens and tokens[0] == self.bos_token:
            tokens = tokens[1:]

        # Now join as before
        if self.dataset_name == 'wsi_report':
            text = ''
            for token in tokens:
                if token in '.,;:':
                    text = text.rstrip() + token + ' '
                else:
                    text += token + ' '
            return text.strip()
        else:
            return ' '.join(tokens)'''



    def decode_batch(self, ids_batch):
        """Convert a batch of token ids back to texts."""
        return [self.decode(ids) for ids in ids_batch]