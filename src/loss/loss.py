import torch.nn as nn
import torch
from typing import Dict, Iterable
from torch import Tensor
import torch.nn.functional as F
from enum import Enum

class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_SIM = lambda x, y: F.cosine_similarity(x, y)
    #COSINE_DISTANCE = lambda x, y: 1-F.cosine_similarity(x, y)

def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """
    def __init__(self, margin, similarity_fct=SiameseDistanceMetric.COSINE_DISTANCE):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.similarity_fct = similarity_fct

    def forward(self, anchor:Tensor, positive:Tensor, negative:Tensor, label=None):

        distance_positive = 1 - self.similarity_fct(anchor, positive)
        distance_negative = 1 - self.similarity_fct(anchor, negative)
        #distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        #distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() 


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss. Similar to ConstrativeLoss, but it selects hard positive (positives that are far apart)
    and hard negative pairs (negatives that are close) and computes the loss only for these pairs. Often yields
    better performances than  ConstrativeLoss.

    :param model: SentenceTransformer model
    :param distance_metric: Function that returns a distance between two emeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used
    :param margin: Negative samples (label == 0) should have a distance of at least the margin value.
    :param size_average: Average by the size of the mini-batch.

    Example::

        from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
        from torch.utils.data import DataLoader

        model = SentenceTransformer('all-MiniLM-L6-v2')
        train_examples = [
            InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
            InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0)]

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
        train_loss = losses.OnlineContrastiveLoss(model=model)

        model.fit([(train_dataloader, train_loss)], show_progress_bar=True)
    """

    def __init__(self, similarity_fct=SiameseDistanceMetric.COSINE_DISTANCE, margin: float = 0.5):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.similarity_fct = similarity_fct

    def forward(self, anchor:Tensor, positive:Tensor, negative:Tensor, label=None):
        
        poss = 1 - self.similarity_fct(anchor, positive)
        negs = 1 - self.similarity_fct(anchor, negative)

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
        loss = positive_loss + negative_loss
        return loss


class MultipleNegativesRankingLoss(nn.Module):
    """
        This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
        where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.

        For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
        n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.

        This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
        as it will sample in each batch n-1 negative docs randomly.

        The performance usually increases with increasing batch sizes.

        For more information, see: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)

        You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
        (a_1, p_1, n_1), (a_2, p_2, n_2)

        Here, n_1 is a hard negative for (a_1, p_1). The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.

        Example::

            from sentence_transformers import SentenceTransformer, losses, InputExample
            from torch.utils.data import DataLoader

            model = SentenceTransformer('distilbert-base-uncased')
            train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
                InputExample(texts=['Anchor 2', 'Positive 2'])]
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
            train_loss = losses.MultipleNegativesRankingLoss(model=model)
    """
    def __init__(self, temperature: float = 0.05, similarity_fct = SiameseDistanceMetric.COSINE_DISTANCE):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(MultipleNegativesRankingLoss, self).__init__()
        self.temperature = temperature
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()


    def forward(self, anchor:Tensor, positive:Tensor, negative:Tensor, label:Tensor):

        concat_norm = torch.cat([anchor, positive, negative], dim=0)

        concat_batch_size = concat_norm.size(0)
        batch_size = anchor.size(0)

        matrix = concat_norm.matmul(concat_norm.transpose(1, 0))
        matrix_mask = torch.eye(concat_batch_size, device=concat_norm.device, dtype=torch.bool)
        matrix = matrix.masked_fill(matrix_mask, float('-inf'))
        matrix = matrix / self.args.temperature
        final_loss = self.cross_entropy_loss(matrix[:batch_size], label)
        return final_loss

    def get_config_dict(self):
        return {'temperature': self.temperature, 'similarity_fct': self.similarity_fct.__name__}
        

