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
            *args,
            **kwargs
    ):
        super(BaseDataset, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.texts = texts
        self.hf_field = hf_field

    def get_tokenized_text(self, idx: int):
        if isinstance(self.texts, list):
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

        return inputs

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

    def __len__(self):
        return len(self.texts)


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

    def __len__(self):
        return len(self.texts)


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

    def __len__(self):
        return len(self.texts)
