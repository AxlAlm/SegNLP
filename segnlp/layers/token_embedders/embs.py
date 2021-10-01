

# segnlp
from typing import Sequence, Union


# pytroch
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

#gensim
from gensim.models import KeyedVectors


#segnlp
from segnlp import utils
from segnlp.resources import vocabs
from segnlp.resources.vocabs import Vocab

class EmbsBase(nn.Module):

    def __init__(self, vocab : Union[str, list]):
        super().__init__()
        self._vocab = getattr(vocabs, vocab)() if isinstance(vocab, str) else Vocab(list)


    def forward(self, 
                input: Sequence,
                lengths : Tensor = None,
                device : Union[str, torch.device] = "cpu"
                ) -> Tensor:

        ids = torch.LongTensor(self._vocab[utils.ensure_numpy(input)]).to(device)

        # gets embeddings
        embs = self.embs(ids)

        if lengths is None:
            return embs

        embs = pad_sequence(
                            torch.split(
                                        embs,
                                        utils.ensure_list(lengths),
                                        ),
                            batch_first = True,
                            padding_value = 0
                            )

        return embs


class Embs(EmbsBase):

    def __init__(self,
                    vocab : str,
                    embedding_dim: int = None,
                    padding_idx: int = 0, 
                    max_norm: float = None, 
                    norm_type: float = 2.0, 
                    scale_grad_by_freq:bool = False, 
                    sparse = False
                    ):
        super().__init__(vocab)
        self.embs = nn.Embedding( 
                                num_embeddings = len(self._vocab), 
                                embedding_dim = embedding_dim, 
                                padding_idx = padding_idx, 
                                max_norm = max_norm, 
                                norm_type = norm_type, 
                                scale_grad_by_freq = scale_grad_by_freq, 
                                sparse = sparse, 
                                )
        self.output_size = self.embs.embedding_dim


class PretrainedEmbs(EmbsBase):

    def __init__(self,
                vocab: Vocab,
                path_to_pretrained: str = None,
                freeze: bool = False,
                ):
        super().__init__(vocab)

        # we create embedding matrix for the given vocab. in the order given
        # then initialize embedding weights weights with the pret
        self.embs = nn.Embedding.from_pretrained(
                                                    self.__create_vocab_matrix(path_to_pretrained), 
                                                    freeze=freeze
                                                    )
        self.output_size = self.embs.embedding_dim


    def __create_vocab_matrix(self, path_to_pretrained:str):

        model = KeyedVectors.load_word2vec_format(path_to_pretrained, binary=True if path_to_pretrained.endswith(".bin") else False)

        vocab_matrix = torch.zeros((len(self._vocab), model.vector_size))

        for i, word in self._vocab._id2item.items():

            if word not in model:
                vocab_matrix[i] = torch.rand(model.vector_size, dtype=torch.float)
                continue
            
            vocab_matrix[i] = torch.tensor(model[word], dtype=torch.float)

    
        return vocab_matrix


