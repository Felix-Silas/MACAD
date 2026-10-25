import torch
import torch.nn.functional as F

class PredictionLoss(object):
    def deep_svdd_loss(self, inputs, labels, center, squared_radius=None):
        """Supervised DeepSVDD objective from Eq. (11).

        The paper directly uses the squared distance to the fixed center in the
        binary entropy objective. Clamping only protects the logarithms from
        numerical infinities; it does not rescale the distance.
        """
        dist_to_center = torch.sum((inputs - center.detach()) ** 2, dim=1)
        labels_src = labels.squeeze().to(inputs.device)
        probability = dist_to_center.clamp(min=1e-6, max=1.0 - 1e-6)
        target = labels_src.float()
        return -(target * torch.log(probability) +
                 (1.0 - target) * torch.log1p(-probability)).mean()

class SupervisedContrastiveLoss(object):
    def __init__(self, dataset_type="smd"):
        self.dataset_type = dataset_type

    def get_sup_cont_tripletloss(self, output_src1, output_pos1, output_neg1, labels_src, margin=1.0): # Triplet Contrastive loss
    
        tau = 1
        # Source contextual pairs are label-aware for both classes; target
        # labels are all zero by construction, so no anchor filtering is needed.
        output_src = output_src1
        output_pos = output_pos1
        output_neg = output_neg1


        if not len(output_src):
            return output_src1.sum() * 0.0

        dist_pos = F.pairwise_distance(output_src, output_pos, 2)
        # Every negative remains paired with its own anchor rather than being
        # pooled across unrelated anchors in the minibatch.
        dist_neg = F.pairwise_distance(output_src, output_neg, 2)

        loss = torch.relu(dist_pos**2 - dist_neg**2 + margin)

        return loss / tau
