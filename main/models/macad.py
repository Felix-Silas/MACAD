import sys
sys.path.append("../..")

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Function


from utils.tcn_no_norm import TemporalConvNet
from utils.mlp import MLP


# Gradient Reversal Layer
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha  # ctx는 Pytorch의 context object로, forward와 backward 연산 간 데이터를 공유하기 위해 사용
        return x.view_as(x)  

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha # grad_output.neg()는 역전파 시 기울기를 반전시키는 역할을 하며, ctx.alpha는 학습률 조정에 사용
        return output, None


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=1):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), # 동일한 차원으로 변하더라도, 선형 변환 및 비선형 활성화 함수로 모델이 더 복잡한 패턴 학습
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
##            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class MACAD_NN(nn.Module):
    def __init__(self, num_inputs, num_channels, projector_hidden_dim=1024,
                 discriminator_hidden_dim=1024, use_batch_norm=False,
                 kernel_size=7, stride=1, dilation_factor=3, dropout=0.1):
        
        super(MACAD_NN, self).__init__()

        # The final TCN channel is the representation. For the four benchmark
        # datasets num_channels is fixed to [128, 256, 512].
        self.encoder_q = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels, kernel_size=kernel_size,
                                         stride=stride, dilation_factor=dilation_factor, dropout=dropout) # 쿼리 데이터를 위한 TCN
        self.representation_dim = num_channels[-1]
        
        # projector for query
        self.projector = MLP(input_dim=self.representation_dim, hidden_dim=projector_hidden_dim,
                             output_dim=self.representation_dim, use_batch_norm=use_batch_norm)

        # Classifier trained by source query
        self.predictor = DeepSVDD(input_dim=self.representation_dim)

        # Discriminator
        self.discriminator = Discriminator(input_dim=self.representation_dim,
                                           hidden_dim=discriminator_hidden_dim,
                                           output_dim=1)

    def _encode(self, sequence):
        features = self.encoder_q(sequence.transpose(1, 2))[:, :, -1]
        return nn.functional.normalize(features, dim=1)



    def forward(self, sequence_q_s, real_s, sequence_q_t, alpha,
                seq_src_positive, seq_src_negative, seq_trg_positive, seq_trg_negative):

        '''
        sequence_q_s: 소스 도메인의 쿼리 시계열 데이터 (NxLxC, 배치 X 시퀀스 길이 X 채널)
        real_s: 소스 도메인의 실제 레이블 (배치 X 1, 0 또는 1)
        '''
       
        # 소스 도메인 쿼리 특징 계산
        q_s = self._encode(sequence_q_s)

        # 타겟 도메인 쿼리 특징 계산
        q_t = self._encode(sequence_q_t)

        # Positive/Negative 쌍 처리 (Nx3xLxC -> con, tem, tran)
        q_s_pos_con = self._encode(seq_src_positive[:, 0])
        q_s_neg_con = self._encode(seq_src_negative[:, 0])

        q_s_pos_tem = self._encode(seq_src_positive[:, 1])
        q_s_neg_tem = self._encode(seq_src_negative[:, 1])

        q_s_pos_tran = self._encode(seq_src_positive[:, 2])
        q_s_neg_tran = self._encode(seq_src_negative[:, 2])

        q_t_pos_con = self._encode(seq_trg_positive[:, 0])
        q_t_neg_con = self._encode(seq_trg_negative[:, 0])

        q_t_pos_tem = self._encode(seq_trg_positive[:, 1])
        q_t_neg_tem = self._encode(seq_trg_negative[:, 1])

        q_t_pos_tran = self._encode(seq_trg_positive[:, 2])
        q_t_neg_tran = self._encode(seq_trg_negative[:, 2])

        # Projector로 projection
        p_q_s = self.projector(q_s)
        p_q_s = nn.functional.normalize(p_q_s, dim=1)
        p_q_s_pos_con = self.projector(q_s_pos_con)
        p_q_s_pos_con = nn.functional.normalize(p_q_s_pos_con, dim=1)
        p_q_s_neg_con = self.projector(q_s_neg_con)
        p_q_s_neg_con = nn.functional.normalize(p_q_s_neg_con, dim=1)
        p_q_s_pos_tem = self.projector(q_s_pos_tem)
        p_q_s_pos_tem = nn.functional.normalize(p_q_s_pos_tem, dim=1)
        p_q_s_neg_tem = self.projector(q_s_neg_tem)
        p_q_s_neg_tem = nn.functional.normalize(p_q_s_neg_tem, dim=1)
        p_q_s_pos_tran = self.projector(q_s_pos_tran)
        p_q_s_pos_tran = nn.functional.normalize(p_q_s_pos_tran, dim=1)
        p_q_s_neg_tran = self.projector(q_s_neg_tran)
        p_q_s_neg_tran = nn.functional.normalize(p_q_s_neg_tran, dim=1)

        p_q_t = self.projector(q_t)
        p_q_t = nn.functional.normalize(p_q_t, dim=1)
        p_q_t_pos_con = self.projector(q_t_pos_con)
        p_q_t_pos_con = nn.functional.normalize(p_q_t_pos_con, dim=1)
        p_q_t_neg_con = self.projector(q_t_neg_con)
        p_q_t_neg_con = nn.functional.normalize(p_q_t_neg_con, dim=1)
        p_q_t_pos_tem = self.projector(q_t_pos_tem)
        p_q_t_pos_tem = nn.functional.normalize(p_q_t_pos_tem, dim=1)
        p_q_t_neg_tem = self.projector(q_t_neg_tem)
        p_q_t_neg_tem = nn.functional.normalize(p_q_t_neg_tem, dim=1)
        p_q_t_pos_tran = self.projector(q_t_pos_tran)
        p_q_t_pos_tran = nn.functional.normalize(p_q_t_pos_tran, dim=1)
        p_q_t_neg_tran = self.projector(q_t_neg_tran)
        p_q_t_neg_tran = nn.functional.normalize(p_q_t_neg_tran, dim=1)

        # DOMAIN DISCRIMINATION
        real_s = real_s.squeeze(1) # Nx1 -> Nx
        q_s_reversed = ReverseLayerF.apply(q_s, alpha) # GRL을 소스 도메인 쿼리 특징에 적용하여 역전파 시 기울기를 반전시킴

        q_t_reversed = ReverseLayerF.apply(q_t, alpha)

        q_reversed = torch.cat([q_s_reversed, q_t_reversed], dim=0)
        
        pred_domain = self.discriminator(q_reversed)


        # 도메인 label
        domain_label_s = torch.ones((len(q_s), 1)).to(device=q_s.device)
        domain_label_t = torch.zeros((len(q_t), 1)).to(device=q_t.device)

        labels_domain = torch.cat([domain_label_s, domain_label_t], dim=0)


        # SOURCE Prediction task
        '''
        pred_s: 소스 도메인 쿼리 특징을 기반으로 예측된 anomaly score
        center: DeepSVDD의 중심 (hypersphere의 중심)
        squared_radius: DeepSVDD의 반지름 (hypersphere의 반지름)
        '''
        pred_s, center, squared_radius = self.predictor(q_s)

        return (pred_domain, labels_domain, pred_s, center, squared_radius,
                q_s, 
                (q_s_pos_con, q_s_pos_tem, q_s_pos_tran), 
                (q_s_neg_con, q_s_neg_tem, q_s_neg_tran),
                p_q_s, 
                (p_q_s_pos_con, p_q_s_pos_tem, p_q_s_pos_tran), 
                (p_q_s_neg_con, p_q_s_neg_tem, p_q_s_neg_tran),
                q_t, 
                (q_t_pos_con, q_t_pos_tem, q_t_pos_tran), 
                (q_t_neg_con, q_t_neg_tem, q_t_neg_tran),
                p_q_t, 
                (p_q_t_pos_con, p_q_t_pos_tem, p_q_t_pos_tran), 
                (p_q_t_neg_con, p_q_t_neg_tem, p_q_t_neg_tran))

    def get_encoding(self, sequence):

        return self._encode(sequence)

    def predict(self, sequence):
        # Get the encoding of a sequence from a given domain
        q = self.get_encoding(sequence['sequence'])

        # Make the prediction based on the encoding
        dist, center, squared_radius = self.predictor(q)

        return dist  # y

class DeepSVDD(nn.Module):
    def __init__(self, input_dim):
        super(DeepSVDD, self).__init__()

        self.input_dim = input_dim
        self.register_buffer("center", torch.zeros(input_dim))
        self.register_buffer("squared_radius", torch.tensor(0.0))
        self.register_buffer("center_initialized", torch.tensor(False))

    @torch.no_grad()
    def initialize_center(self, normal_features: torch.Tensor):
        """Initialize the fixed center with the source-normal feature mean."""
        if normal_features.dim() == 3:
            normal_features = normal_features.mean(dim=1)
        if not len(normal_features):
            raise ValueError("Cannot initialize DeepSVDD center without normal features")
        self.center.copy_(normal_features.mean(dim=0))
        self.center_initialized.fill_(True)

    def forward(self, x):
        """
        x: (batch_size, feature_dim) - TCN encoder의 출력 (normalized)
        """
        if x.dim() > 2:
            x = x.mean(dim=1) if x.dim() == 3 else x.squeeze()

        dist = torch.sum((x - self.center) ** 2, dim=1)
        return dist, self.center, self.squared_radius
