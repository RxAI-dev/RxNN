import torch
from typing import TypedDict

class SpecialTokenIds(TypedDict):
    bos: int
    eos: int
    pad: int

class TokenizedDict(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

def smart_concat_critic_states(
        prev_query: TokenizedDict,
        prev_answer: TokenizedDict,
        next_query: TokenizedDict,
        max_length: int,
        special_token_ids: SpecialTokenIds
) -> TokenizedDict:
    """
    Smart vectorized concatenation of MRL critic states - previous interaction (query and answer) and next query.
    It creates a batch of critic input sequences from previous query, previous answer and next query batches.
    Used in MRL to concatenate critic states in correct format.

    All the concatenated sequences (batches) are padded to the same max length, but the result should have two times
    longer max length. Single max length is made to fit single query and answer, but here we have additional next query,
    so we are using 2x longer sequence for safety.

    Args:
        prev_query (TokenizedDict): Batch of tokenized queries with attention masks from previous interaction
        prev_answer (TokenizedDict): Batch of tokenized answers with attention masks from previous interaction
        next_query (TokenizedDict): Batch of tokenized queries with attention masks from next interaction
        max_length (int): Max length of result sequence.
        special_token_ids (SpecialTokenIds): Indexes of required special tokens: BOS, EOS, PAD
    """
    device = prev_query['input_ids'].device
    batch_size = prev_query['input_ids'].size(0)

    # Get special token ids
    eos_token = special_token_ids['eos']
    bos_token = special_token_ids['bos']
    pad_token = special_token_ids['pad']

    # Get actual lengths
    query_lens = prev_query['attention_mask'].sum(dim=1)
    answer_lens = prev_answer['attention_mask'].sum(dim=1)
    next_query_lens = next_query['attention_mask'].sum(dim=1)

    # Create position grid [batch_size, max_length]
    positions = torch.arange(max_length, device=device).expand(batch_size, -1)

    # Calculate section boundaries
    section1_end = query_lens.unsqueeze(1)
    section2_end = section1_end + answer_lens.unsqueeze(1)
    section3_end = section2_end + 2  # For EOS+BOS
    section4_end = section3_end + next_query_lens.unsqueeze(1)

    # Create masks for each section
    mask_prev_query = positions < section1_end
    mask_prev_answer = (positions >= section1_end) & (positions < section2_end)
    mask_eos = positions == section2_end
    mask_bos = positions == section2_end + 1
    mask_next_query = (positions >= section3_end) & (positions < section4_end)

    # Build combined_ids
    combined_ids = torch.full((batch_size, max_length), pad_token, device=device)

    # Fill sections using vectorized operations
    combined_ids = torch.where(
        mask_prev_query,
        prev_query['input_ids'].gather(1, positions.clamp(max=section1_end - 1)),
        combined_ids
    )

    combined_ids = torch.where(
        mask_prev_answer,
        prev_answer['input_ids'].gather(1, (positions - section1_end).clamp(min=0)),
        combined_ids
    )

    combined_ids = torch.where(
        mask_eos,
        eos_token,
        combined_ids
    )

    combined_ids = torch.where(
        mask_bos,
        bos_token,
        combined_ids
    )

    combined_ids = torch.where(
        mask_next_query,
        next_query['input_ids'].gather(1, (positions - section3_end).clamp(min=0)),
        combined_ids
    )

    # Build attention mask
    combined_mask = (positions < section4_end).long()

    return {
        'input_ids': combined_ids,
        'attention_mask': combined_mask
    }


def smart_concat(query: TokenizedDict, answer: TokenizedDict, max_length: int, pad_token_id: int) -> TokenizedDict:
    """
    Smart vectorized concatenation of interaction parts - query and answer. It creates
    batch of interactions from query and answer batches. Used in MRL to concatenate data
    to encode and update memory.

    Query and answer sequences are padded to the same max length, and the result also has
    the same length.

    Args:
        query (TokenizedDict): Batch of tokenized queries with attention masks
        answer (TokenizedDict): Batch of tokenized answers with attention masks
        max_length (int): Max length of each sequence - query, answer and result.
        pad_token_id (int): Index of padding token
    """
    device = query['input_ids'].device
    batch_size = query['input_ids'].size(0)

    # Get actual lengths from attention masks
    query_lens = query['attention_mask'].sum(dim=1)
    answer_lens = answer['attention_mask'].sum(dim=1)

    # Create combined length tensor
    combined_lens = torch.minimum(query_lens + answer_lens,
                                  torch.full_like(query_lens, max_length))

    # Create position indices [batch_size, max_length]
    positions = torch.arange(max_length, device=device).expand(batch_size, -1)

    # Create mask for query/answer parts
    query_mask = positions < query_lens.unsqueeze(1)
    answer_mask = (positions >= query_lens.unsqueeze(1)) & (positions < combined_lens.unsqueeze(1))

    # Calculate answer positions with overflow protection
    answer_pos = (positions - query_lens.unsqueeze(1)).clamp(min=0)

    # Build combined_ids using vectorized where
    combined_ids = torch.where(
        query_mask,
        query['input_ids'].gather(1, torch.minimum(positions, query_lens.unsqueeze(1) - 1)),
        torch.where(
            answer_mask,
            answer['input_ids'].gather(1, answer_pos),
            query['input_ids'].new_full((1,), pad_token_id)
        )
    )

    # Build attention mask
    combined_mask = (positions < combined_lens.unsqueeze(1)).long()

    return {
        'input_ids': combined_ids,
        'attention_mask': combined_mask
    }
