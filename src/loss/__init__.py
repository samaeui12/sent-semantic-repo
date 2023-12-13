from loss.loss import MultipleNegativesRankingLoss, TripletLoss, OnlineContrastiveLoss

Loss_MAPPING_DICT = {
    'MultipleNegativesRankingLoss': MultipleNegativesRankingLoss,
    'TripletLoss': TripletLoss,
    'OnlineContrastiveLoss': OnlineContrastiveLoss
}
