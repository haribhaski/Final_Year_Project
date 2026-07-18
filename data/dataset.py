from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class WikiText103DocumentDataset(Dataset):
    """
    Loads WikiText-103 while preserving document boundaries.

    Each item represents one complete Wikipedia article divided into
    sequential token chunks.
    """

    FILE_NAMES = {
        "train": "wiki.train.tokens",
        "validation": "wiki.valid.tokens",
        "test": "wiki.test.tokens",
    }

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        split: str = "train",
        chunk_size: int = 256,
        min_document_tokens: int = 64,
        max_documents: int | None = None,
    ) -> None:
        if split not in self.FILE_NAMES:
            raise ValueError(
                f"split must be one of {list(self.FILE_NAMES)}, got {split}"
            )

        if chunk_size < 2:
            raise ValueError("chunk_size must be at least 2.")

        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.split = split
        self.chunk_size = chunk_size
        self.min_document_tokens = min_document_tokens

        file_path = self.data_dir / self.FILE_NAMES[split]

        if not file_path.exists():
            raise FileNotFoundError(
                f"WikiText file not found: {file_path}"
            )

        raw_documents = self._read_documents(file_path)

        if max_documents is not None:
            raw_documents = raw_documents[:max_documents]

        self.documents = self._tokenize_documents(raw_documents)

        if not self.documents:
            raise RuntimeError(
                "No usable documents were found after preprocessing."
            )

        print(
            f"Loaded {len(self.documents)} documents "
            f"from {file_path}"
        )

    def _read_documents(
        self,
        file_path: Path,
    ) -> List[Dict[str, str]]:
        """
        WikiText articles usually begin with a line such as:

            = Article Title =
        """

        documents: List[Dict[str, str]] = []

        current_title = "untitled"
        current_lines: List[str] = []

        with file_path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:
            for raw_line in file:
                line = raw_line.strip()

                if not line:
                    continue

                is_title = (
                    line.startswith("= ")
                    and line.endswith(" =")
                    and not line.startswith("==")
                )

                if is_title:
                    if current_lines:
                        documents.append(
                            {
                                "title": current_title,
                                "text": "\n".join(current_lines),
                            }
                        )

                    current_title = line.strip("= ").strip()
                    current_lines = []
                else:
                    current_lines.append(line)

        if current_lines:
            documents.append(
                {
                    "title": current_title,
                    "text": "\n".join(current_lines),
                }
            )

        return documents

    def _tokenize_documents(
        self,
        raw_documents: List[Dict[str, str]],
    ) -> List[Dict[str, object]]:
        processed: List[Dict[str, object]] = []

        eos_token_id = self.tokenizer.eos_token_id

        for document_id, document in enumerate(raw_documents):
            token_ids = self.tokenizer.encode(
                document["text"],
                add_special_tokens=False,
            )

            if eos_token_id is not None:
                token_ids.append(eos_token_id)

            if len(token_ids) < self.min_document_tokens:
                continue

            chunks: List[torch.Tensor] = []

            for start in range(0, len(token_ids), self.chunk_size):
                chunk_ids = token_ids[
                    start : start + self.chunk_size
                ]

                # At least two tokens are required for next-token loss.
                if len(chunk_ids) < 2:
                    continue

                chunks.append(
                    torch.tensor(
                        chunk_ids,
                        dtype=torch.long,
                    )
                )

            if chunks:
                processed.append(
                    {
                        "document_id": document_id,
                        "title": document["title"],
                        "num_tokens": len(token_ids),
                        "chunks": chunks,
                    }
                )

        return processed

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.documents[index]


def document_collate_fn(
    batch: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    Initially use batch_size=1 because every document contains a different
    number of chunks.
    """

    if len(batch) != 1:
        raise ValueError(
            "Use batch_size=1 with document_collate_fn."
        )

    return batch[0]