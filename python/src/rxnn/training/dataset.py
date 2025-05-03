import torch
from torch.utils.data import Dataset
from datasets import Dataset as HfDataset, load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from .tokenizer import load_tokenizer_from_hf_hub

from typing import Union


class BaseDataset(Dataset):
    def __init__(
            self,
            texts: Union[list[str], HfDataset],
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = 1024,
            hf_field: str = 'text',
            cache_tokenized: bool = False,
            cache_remove_text: bool = True,
            tokenize_in_background: bool = False,
            batch_size: int = 1,
            *args,
            **kwargs
    ):
        super(BaseDataset, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.texts = texts
        self.hf_field = hf_field
        self.is_pre_tokenized = False
        self.cache_tokenized = cache_tokenized
        self.cache_remove_text = cache_remove_text
        self.inputs = []
        self.is_txt_list = isinstance(self.texts, list)
        self.tokenize_in_background = tokenize_in_background
        self.bg_next = []
        self.bg_queue = None
        self.batch_size = batch_size
        self.last_idx = 0
        if tokenize_in_background:
            for i in range(self.batch_size):
                self.bg_next.append(self.get_tokenized_text(i))
            self.last_idx = self.batch_size - 1


    def __len__(self):
        return len(self.texts if not self.is_pre_tokenized else self.inputs)

    def get_tokenized_text(self, idx: int, txt: str = None):
        if self.is_pre_tokenized:
            return self.inputs[idx]
        elif self.tokenize_in_background:
            if idx == self.last_idx - self.batch_size:
                if self.bg_queue is not None:
                    self.bg_next = self.bg_queue
                    self.bg_queue = None
                # TODO: schedule tokenizing a batch in background
            elif idx == self.last_idx:
                item = self.bg_next[idx]
                self.bg_next = []
                return item

            if idx <= self.last_idx:
                if self.bg_queue is not None:
                    self.bg_next = self.bg_queue
                    self.bg_queue = None

                new_idx = idx - (self.last_idx - self.batch_size)
                if new_idx in self.bg_next:
                    return self.bg_next[new_idx]
                else:
                    if self.is_txt_list:
                        text = self.texts[idx]
                    else:
                        text = self.texts[idx][self.hf_field]

                    inputs = self.tokenizer(
                        text,
                        max_length=self.max_seq_len,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt',
                        return_attention_mask=True
                    )
                    if not (inputs['input_ids'][0] < self.tokenizer.vocab_size).all():
                        inputs['input_ids'][0][
                            (inputs['input_ids'][0] >= self.tokenizer.vocab_size)] = self.tokenizer.unk_token_id
                    if not (inputs['input_ids'][0] >= 0).all():
                        inputs['input_ids'][0][inputs['input_ids'][0] < 0] = self.tokenizer.unk_token_id

                    return inputs
        else:
            if txt is not None:
                text = txt
            elif self.is_txt_list:
                text = self.texts[idx]
            else:
                text = self.texts[idx][self.hf_field]

            inputs = self.tokenizer(
                text,
                max_length=self.max_seq_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
                return_attention_mask=True
            )
            if not (inputs['input_ids'][0] < self.tokenizer.vocab_size).all():
                inputs['input_ids'][0][(inputs['input_ids'][0] >= self.tokenizer.vocab_size)] = self.tokenizer.unk_token_id
            if not (inputs['input_ids'][0] >= 0).all():
                inputs['input_ids'][0][inputs['input_ids'][0] < 0] = self.tokenizer.unk_token_id

            if self.cache_tokenized:
                self.inputs.append(inputs)
                if len(self.inputs) == len(self.texts):
                    self.is_pre_tokenized = True
                    if self.cache_remove_text:
                        del self.texts
                        self.texts = None

            return inputs

    def get_subset(self, size: float, from_start: bool = False, **kwargs) -> "BaseDataset":
        split_point = int(len(self.texts) * ((1 - size) if not from_start else size))
        if not isinstance(self.texts, list):
            subset = self.texts.select(range(split_point, len(self.texts)) if not from_start else range(split_point))
            self.texts = self.texts.select(range(split_point) if not from_start else range(split_point, len(self.texts)))
        else:
            subset = self.texts[split_point:-1] if not from_start else self.texts[0:split_point]
            self.texts = self.texts[0:split_point] if not from_start else self.texts[split_point:-1]
        return self.__class__(subset, self.tokenizer, max_seq_len=self.max_seq_len, hf_field=self.hf_field, **kwargs)

    def pre_tokenize(self, verbose: bool = False, log_interval: int = 10_000, map_hf_ds_to_list: bool = True):
        """
        Pre-tokenizes all the items in the dataset, for faster training. Training with pre-tokenized
        dataset could be even 2x faster.

        !! This method has extremely high memory usage, when used with HuggingFace datasets,
        because of converting it to list. Additionally, for the most optimal performance,
        pre-tokenized items are in reversed order - it shouldn't matter for training, as
        items are shuffled then by DataLoader, but you should keep that in mind in case
        of reproducibility.

        :param(bool) verbose:
        :param(int) log_interval: Interval of verbose logs
        """
        if not self.is_pre_tokenized:
            num_texts = len(self.texts)
            txts = self.texts if self.is_txt_list else self.texts.to_list()
            del self.texts
            self.texts = None
            for index in range(num_texts):
                item = txts.pop() if self.is_txt_list else txts.pop()[self.hf_field]
                self.inputs.append(self.get_tokenized_text(index, txt=item))
                if verbose and index % log_interval == 0:
                    print(f'Processed {index + 1}/{num_texts}')
            self.is_pre_tokenized = True


    @classmethod
    def from_hf_hub(
            cls,
            dataset_id: str,
            subset: str = None,
            split: str = 'train',
            target_field: str = 'text',
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            tokenizer_hub_id: str = None,
            max_seq_len: int = 1024,
            load_kwargs: dict = None,
            load_tokenizer_kwargs: dict = None,
            **kwargs
    ):
        """
        Load dataset from HuggingFace Hub and convert it to RxNN training dataset.

        One of the `tokenizer` or `tokenizer_hub_id` args must be provided. If both are provided, `tokenizer` will be used.

        Args:
            dataset_id (str): Hub dataset repository name
            subset (str): Dataset subset
            split (str): Dataset split (default: "train")
            target_field (str): Name of dataset field used for training (default: "text")
            tokenizer (PreTrainedTokenizer): HuggingFace Tokenizer used for training (default: None)
            tokenizer_hub_id (str): HuggingFace Hub ID of tokenizer to load (default: None)
            max_seq_len (int): Maximum sequence length for training (default: 1024)
            load_kwargs (dict): Additional args for HuggingFace API load_dataset function
            load_tokenizer_kwargs (dict): Additional args for loading tokenizer from HuggingFace API with `huggingface_hub.hf_hub_download`
            **kwargs: Additional args for RxNN Dataset class
        """
        assert tokenizer is not None or tokenizer_hub_id is not None, "One of the `tokenizer` or `tokenizer_hub_id` args must be provided."

        if tokenizer is None:
            tokenizer = load_tokenizer_from_hf_hub(tokenizer_hub_id, **load_tokenizer_kwargs)

        hf_dataset = load_dataset(dataset_id, subset, split=split, **load_kwargs)

        return cls(hf_dataset, tokenizer, max_seq_len=max_seq_len, hf_field=target_field, **kwargs)

    @classmethod
    def concat_from_hf_hub(
            cls,
            dataset_ids: tuple[str],
            subsets: tuple[str] = None,
            split: str = 'train',
            target_field: str = 'text',
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            tokenizer_hub_id: str = None,
            max_seq_len: int = 1024,
            load_kwargs: dict = None,
            load_tokenizer_kwargs: dict = None,
            **kwargs
    ):
        """
        Load and concatenate multiple datasets from HuggingFace Hub and convert them to RxNN training dataset.
        All datasets should use the same split and target field. If it's not the case, just use `load_dataset` and pass the
        result to RxNN dataset constructor directly.

        One of the `tokenizer` or `tokenizer_hub_id` args must be provided. If both are provided, `tokenizer` will be used.

        Args:
            dataset_ids (tuple[str]): Hub dataset repository names
            subsets (tuple[str]): Dataset subsets (default: None)
            split (str): Dataset split (default: "train")
            target_field (str): Name of dataset field used for training (default: "text")
            tokenizer (PreTrainedTokenizer): HuggingFace Tokenizer used for training (default: None)
            tokenizer_hub_id (str): HuggingFace Hub ID of tokenizer to load (default: None)
            max_seq_len (int): Maximum sequence length for training (default: 1024)
            load_kwargs (dict): Additional args for HuggingFace API load_dataset function
            load_tokenizer_kwargs (dict): Additional args for loading tokenizer from HuggingFace API with `huggingface_hub.hf_hub_download`
            **kwargs: Additional args for RxNN Dataset class
        """
        assert tokenizer is not None or tokenizer_hub_id is not None, "One of the `tokenizer` or `tokenizer_hub_id` args must be provided."

        if tokenizer is None:
            tokenizer = load_tokenizer_from_hf_hub(tokenizer_hub_id, **load_tokenizer_kwargs)

        hf_datasets = [
            load_dataset(dataset_id, subset, split=split, **load_kwargs) for dataset_id, subset in zip(dataset_ids, subsets)
        ] if subsets is not None else [
            load_dataset(dataset_id, split=split, **load_kwargs) for dataset_id in dataset_ids
        ]
        hf_dataset = concatenate_datasets(hf_datasets)

        return cls(hf_dataset, tokenizer, max_seq_len=max_seq_len, hf_field=target_field, **kwargs)



class JointLMDataset(BaseDataset):
    def __init__(
            self,
            texts: Union[list[str], HfDataset],
            tokenizer: PreTrainedTokenizer,
            max_seq_len: int = 1024,
            mask_prob: float = 0.15,
            hf_field: str = 'text',
            *args,
            **kwargs
    ):
        super(JointLMDataset, self).__init__(texts, tokenizer, max_seq_len, hf_field, *args, **kwargs)
        self.mask_prob = mask_prob

    def __getitem__(self, idx: int) -> dict[str, dict[str, torch.Tensor]]:
        inputs = self.get_tokenized_text(idx)
        encoder_input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        decoder_input_ids = encoder_input_ids.clone()

        encoder_labels = encoder_input_ids.clone()
        decoder_targets = encoder_input_ids.clone()

        # Create masked indices
        masked_indices = torch.bernoulli(
            torch.full(encoder_labels.shape, self.mask_prob)
        ).bool() & attention_mask.bool()

        # Apply mask
        encoder_labels[~masked_indices] = -100
        encoder_input_ids[masked_indices] = self.tokenizer.mask_token_id

        return {
            'decoder': {
                'input_ids': decoder_input_ids,
                'targets': decoder_targets,
            },
            'encoder': {
                'input_ids': encoder_input_ids,
                'labels': encoder_labels,
            },
            'attention_mask': attention_mask,
        }


class MaskedLMDataset(BaseDataset):
    def __init__(
            self,
            texts: Union[list[str], HfDataset],
            tokenizer: PreTrainedTokenizer,
            max_seq_len: int = 1024,
            mask_prob: float = 0.15,
            hf_field: str = 'text',
            *args,
            **kwargs
    ):
        super(MaskedLMDataset, self).__init__(texts, tokenizer, max_seq_len, hf_field, *args, **kwargs)
        self.mask_prob = mask_prob

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        inputs = self.get_tokenized_text(idx)

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        labels = input_ids.clone()

        # Create masked indices
        masked_indices = torch.bernoulli(
            torch.full(labels.shape, self.mask_prob)
        ).bool() & attention_mask.bool()

        # Apply mask
        labels[~masked_indices] = -100
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class AutoregressiveLMDataset(BaseDataset):
    def __init__(
            self,
            texts: Union[list[str], HfDataset],
            tokenizer: PreTrainedTokenizer,
            max_seq_len: int = 1024,
            hf_field: str = 'text',
            *args,
            **kwargs
    ):
        super(AutoregressiveLMDataset, self).__init__(texts, tokenizer, max_seq_len, hf_field, *args, **kwargs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        inputs = self.get_tokenized_text(idx)

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        targets = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'targets': targets
        }
