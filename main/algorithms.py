import sys
sys.path.append("..")
import os
import numpy as np
import torch
import torch.nn as nn
from utils.util_progress_log import AverageMeter, PredictionMeter, get_dataset_type
from utils.loss import PredictionLoss, SupervisedContrastiveLoss
from models.macad import MACAD_NN
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Fixed architecture shared by MSL, SMAP, SMD, and Boiler.
TCN_CHANNELS = [128, 256, 512]
PROJECTOR_HIDDEN_DIM = 1024
ADAPTIVE_WEIGHT_HIDDEN_DIM = 256
DISCRIMINATOR_HIDDEN_DIM = 1024


# MLP 기반 가중치 예측 모듈
class WeightPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=3, init_temperature=2.0):
        super(WeightPredictor, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim)  # con, tem, tran에 대한 3개의 가중치 출력
        )

        nn.init.zeros_(self.mlp[-1].bias)

        self.register_buffer("temperature", torch.tensor(float(init_temperature)))

    def forward(self, x, progress=0.0):
        logits = self.mlp(x)
        
        weights = F.softmax(logits / self.temperature, dim=-1)

        # Minimizing negative entropy maximizes entropy and prevents collapse.
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean()
        neg_entropy_loss = -entropy
        
        return weights, self.temperature, neg_entropy_loss


#Given the args, it will return the algorithm directly
def get_algorithm(args, input_channels_dim):
    if args.algo_name == "macad":
        return MACAD(args, input_channels_dim)
    raise ValueError(f"Unsupported algorithm: {args.algo_name}")


def get_num_channels(args):
    channels = list(map(int, args.num_channels_TCN.split("-")))
    if channels != TCN_CHANNELS:
        raise ValueError(
            f"MACAD uses fixed TCN channels {TCN_CHANNELS}, got {channels}"
        )
    return channels


class Base_Algorithm(nn.Module):
    def __init__(self, args):
        super(Base_Algorithm, self).__init__()

        self.args = args
        self.algo_name = args.algo_name
        self.dataset_type = get_dataset_type(args)

        self.pred_loss = PredictionLoss()
        self.sup_cont_loss = SupervisedContrastiveLoss(self.dataset_type)

        self.num_channels = get_num_channels(args)

        # AUPRC or Macro F1
        self.main_pred_metric = ""
        if self.dataset_type in ["smd", "msl", "boiler", "smap"]:
            self.main_pred_metric = "roc_auc"
        else:
            self.main_pred_metric = "mac_f1"

        self.init_pred_meters_val()

    def init_pred_meters_val(self):
        self.pred_meter_val_src = PredictionMeter(self.args)
        self.pred_meter_val_trg = PredictionMeter(self.args)

    def init_score_pred(self):
        if self.dataset_type in ["smd", "msl", "boiler", "smap"]:
            return AverageMeter('ROC AUC', ':6.2f')
        else:
            return AverageMeter('Macro F1', ':6.2f')

# MACAD Algorithm
class MACAD(Base_Algorithm):

    def __init__(self, args, input_channels_dim):
        super(MACAD, self).__init__(args)

        self.input_channels_dim = input_channels_dim
        self.model = MACAD_NN(
            num_inputs=input_channels_dim,
            num_channels=self.num_channels,
            projector_hidden_dim=PROJECTOR_HIDDEN_DIM,
            discriminator_hidden_dim=DISCRIMINATOR_HIDDEN_DIM,
            use_batch_norm=args.use_batch_norm,
            kernel_size=args.kernel_size_TCN,
            stride=args.stride_TCN,
            dilation_factor=args.dilation_factor_TCN,
            dropout=args.dropout
        )

        input_dim = self.num_channels[-1]
        self.weight_predictor = WeightPredictor(
            input_dim=input_dim,
            hidden_dim=ADAPTIVE_WEIGHT_HIDDEN_DIM,
            init_temperature=getattr(args, "adaptive_temperature", 1.0),
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.weight_predictor.to(device)
        self.to(device)

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.weight_predictor.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        self.weight_step_history = {
            'step': [],
            'con': [],
            'tem': [],
            'tran': [],
            'neg_entropy': []
        }

   
        self.init_metrics()

    def step(self, sample_batched_src, sample_batched_trg, **kwargs):
        count_step = kwargs.get("count_step", 0)

        p = float(count_step) / 1000
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        seq_q_src, seq_q_trg = sample_batched_src['sequence'], sample_batched_trg['sequence']

        output_disc, target_disc, pred_s, center, squared_radius, \
        q_s_repr, (q_s_pos_con, q_s_pos_tem, q_s_pos_tran), (q_s_neg_con, q_s_neg_tem, q_s_neg_tran), \
        p_q_s, (p_q_s_pos_con, p_q_s_pos_tem, p_q_s_pos_tran), (p_q_s_neg_con, p_q_s_neg_tem, p_q_s_neg_tran), \
        q_t, (q_t_pos_con, q_t_pos_tem, q_t_pos_tran), (q_t_neg_con, q_t_neg_tem, q_t_neg_tran), \
        p_q_t, (p_q_t_pos_con, p_q_t_pos_tem, p_q_t_pos_tran), (p_q_t_neg_con, p_q_t_neg_tem, p_q_t_neg_tran) = \
            self.model(
                sample_batched_src['sequence'], sample_batched_src['label'],
                sample_batched_trg['sequence'], alpha,
                sample_batched_src['positive'], sample_batched_src['negative'],
                sample_batched_trg['positive'], sample_batched_trg['negative']
            )

        loss_disc = F.binary_cross_entropy_with_logits(output_disc, target_disc)

        src_cls_loss = self.pred_loss.deep_svdd_loss(q_s_repr, sample_batched_src['label'], center, squared_radius)

        src_sup_cont_loss_con = self.sup_cont_loss.get_sup_cont_tripletloss(p_q_s, p_q_s_pos_con, p_q_s_neg_con, sample_batched_src['label'], margin=2)
        src_sup_cont_loss_tem = self.sup_cont_loss.get_sup_cont_tripletloss(p_q_s, p_q_s_pos_tem, p_q_s_neg_tem, sample_batched_src['label'], margin=2)
        src_sup_cont_loss_tran = self.sup_cont_loss.get_sup_cont_tripletloss(p_q_s, p_q_s_pos_tran, p_q_s_neg_tran, sample_batched_src['label'], margin=2)

        # 가중치 예측 (KL 없이 negative entropy만)
        weights, current_temp, neg_entropy_loss = self.weight_predictor(q_s_repr)
        w_con, w_tem, w_tran = weights.mean(dim=0)

        if count_step % 10 == 0 and self.training:
            self.weight_step_history['step'].append(count_step)
            self.weight_step_history['con'].append(w_con.item())
            self.weight_step_history['tem'].append(w_tem.item())
            self.weight_step_history['tran'].append(w_tran.item())
            self.weight_step_history['neg_entropy'].append(neg_entropy_loss.item())


        src_sup_cont_loss_total = (
            weights[:, 0] * src_sup_cont_loss_con +
            weights[:, 1] * src_sup_cont_loss_tem +
            weights[:, 2] * src_sup_cont_loss_tran
        ).mean()

        trg_fake_labels = np.zeros(len(sample_batched_trg['label']))

        trg_sup_cont_loss_con = self.sup_cont_loss.get_sup_cont_tripletloss(p_q_t, p_q_t_pos_con, p_q_t_neg_con, trg_fake_labels, margin=2)
        trg_sup_cont_loss_tem = self.sup_cont_loss.get_sup_cont_tripletloss(p_q_t, p_q_t_pos_tem, p_q_t_neg_tem, trg_fake_labels, margin=2)
        trg_sup_cont_loss_tran = self.sup_cont_loss.get_sup_cont_tripletloss(p_q_t, p_q_t_pos_tran, p_q_t_neg_tran, trg_fake_labels, margin=2)

        trg_sup_cont_loss_total = (
            weights[:, 0] * trg_sup_cont_loss_con +
            weights[:, 1] * trg_sup_cont_loss_tem +
            weights[:, 2] * trg_sup_cont_loss_tran
        ).mean()

        # KL loss 완전 제거 → negative entropy loss만 적용

        loss = self.args.weight_loss_disc * loss_disc + \
               self.args.weight_loss_pred * src_cls_loss + \
               self.args.weight_loss_src_sup * src_sup_cont_loss_total + \
               self.args.weight_loss_trg_ccl * trg_sup_cont_loss_total + \
               self.args.weight_loss_entropy * neg_entropy_loss

        self.losses_sup_con.update(src_sup_cont_loss_con.mean().item(), seq_q_src.size(0))
        self.losses_sup_tem.update(src_sup_cont_loss_tem.mean().item(), seq_q_src.size(0))
        self.losses_sup_tran.update(src_sup_cont_loss_tran.mean().item(), seq_q_src.size(0))
        self.losses_trg_con.update(trg_sup_cont_loss_con.mean().item(), seq_q_trg.size(0))
        self.losses_trg_tem.update(trg_sup_cont_loss_tem.mean().item(), seq_q_trg.size(0))
        self.losses_trg_tran.update(trg_sup_cont_loss_tran.mean().item(), seq_q_trg.size(0))

        if self.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.losses_sup.update(src_sup_cont_loss_total.item(), seq_q_src.size(0))
        self.losses_trg.update(trg_sup_cont_loss_total.item(), seq_q_trg.size(0))

        self.losses_disc.update(loss_disc.item(), output_disc.size(0))

        self.losses_pred.update(src_cls_loss.item(), seq_q_src.size(0))

        pred_meter_src = PredictionMeter(self.args)
        pred_meter_src.update(sample_batched_src['label'], pred_s)
        metrics_pred_src = pred_meter_src.get_metrics()
        self.score_pred.update(metrics_pred_src[self.main_pred_metric], sample_batched_src['sequence'].size(0))

        self.losses.update(loss.item(), sample_batched_src['sequence'].size(0))

        if not self.training:
            labels = sample_batched_src.get('point_label', sample_batched_src['label'])
            self.pred_meter_val_src.update(labels, pred_s)
            pred_t = self.model.predict(sample_batched_trg)
            labels = sample_batched_trg.get('point_label', sample_batched_trg['label'])
            self.pred_meter_val_trg.update(labels, pred_t)

    def init_metrics(self):
        self.losses_sup = AverageMeter('L_Src_Sup', ':.4e')
        self.losses_trg = AverageMeter('L_Trg_CCL', ':.4e')
        self.losses_disc = AverageMeter('Loss Disc', ':.4e')
        self.losses_pred = AverageMeter('Loss Pred', ':.4e')
        self.score_pred = self.init_score_pred()
        self.losses = AverageMeter('Loss TOTAL', ':.4e')

        self.losses_sup_con = AverageMeter('L_Src_Sup_Con', ':.4e')
        self.losses_sup_tem = AverageMeter('L_Src_Sup_Tem', ':.4e')
        self.losses_sup_tran = AverageMeter('L_Src_Sup_Tran', ':.4e')
        self.losses_trg_con = AverageMeter('L_Trg_CCL_Con', ':.4e')
        self.losses_trg_tem = AverageMeter('L_Trg_CCL_Tem', ':.4e')
        self.losses_trg_tran = AverageMeter('L_Trg_CCL_Tran', ':.4e')

    def return_metrics(self):
        return [self.losses_sup, self.losses_sup_con, self.losses_sup_tem, self.losses_sup_tran,
                self.losses_trg, self.losses_trg_con, self.losses_trg_tem, self.losses_trg_tran,
                self.losses_disc, self.losses_pred, self.losses]

    def save_state(self, experiment_folder_path):
        torch.save({
            'state_dict': self.model.state_dict(),
            'weight_predictor_state_dict': self.weight_predictor.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, os.path.join(experiment_folder_path, "model_best.pth.tar"))

    def load_state(self, experiment_folder_path):
        checkpoint = torch.load(
            os.path.join(experiment_folder_path, "model_best.pth.tar"),
            map_location=next(self.parameters()).device,
        )
        self.model.load_state_dict(checkpoint['state_dict'])
        self.weight_predictor.load_state_dict(checkpoint['weight_predictor_state_dict'])

    def predict_trg(self, sample_batched):
        y_pred_trg = self.model.predict(sample_batched)
        labels = sample_batched.get('point_label', sample_batched['label'])
        self.pred_meter_val_trg.update(labels, y_pred_trg)

    def predict_src(self, sample_batched):
        y_pred_src = self.model.predict(sample_batched)
        labels = sample_batched.get('point_label', sample_batched['label'])
        self.pred_meter_val_src.update(labels, y_pred_src)

    @torch.no_grad()
    def initialize_center(self, dataloader):
        """Initialize the fixed DeepSVDD center from source-normal features."""
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        feature_sum = None
        normal_count = 0

        try:
            for batch in dataloader:
                sequence = batch['sequence'].to(device, non_blocking=True)
                labels = batch['label'].reshape(-1).to(device)
                normal_features = self.model.get_encoding(sequence)[labels == 0]
                if normal_features.numel() == 0:
                    continue

                batch_sum = normal_features.sum(dim=0)
                feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
                normal_count += normal_features.size(0)
        finally:
            self.train(was_training)

        if normal_count == 0:
            raise ValueError("Source-train contains no normal samples for DeepSVDD center")

        # c is computed exactly once before training. DeepSVDD stores it as a
        # buffer, so it follows the model device/checkpoints but is never
        # included in gradient updates.
        self.model.predictor.initialize_center((feature_sum / normal_count).unsqueeze(0))

    def get_embedding(self, sample_batched):
        feat = self.model.get_encoding(sample_batched['sequence'])
        return feat

    def save_weight_realtime_plot(self, save_dir="plots", exp_name=None):
        if len(self.weight_step_history['step']) == 0:
            print("No step-wise weight history recorded.")
            return

        os.makedirs(save_dir, exist_ok=True)
        steps = self.weight_step_history['step']

        all_weights = []
        all_weights.extend(self.weight_step_history['con'])
        all_weights.extend(self.weight_step_history['tem'])
        all_weights.extend(self.weight_step_history['tran'])

        data_min = min(all_weights)
        data_max = max(all_weights)

        # 여유분 0.05 추가 (너무 좁지 않게 최소 0.0 ~ 1.0 범위는 유지)
        y_min = max(0.0, data_min - 0.05)
        y_max = min(1.0, data_max + 0.05)

        plt.figure(figsize=(14, 7))
        plt.axhline(y=1/3, color='r', linestyle='--', linewidth=1, label='Uniform', alpha=0.3)

        plt.plot(steps, self.weight_step_history['con'], 'o-', label='Contextual', linewidth=2.5, markersize=4, alpha=0.9)
        plt.plot(steps, self.weight_step_history['tem'], 's-', label='Temporal', linewidth=2.5, markersize=4, alpha=0.9)
        plt.plot(steps, self.weight_step_history['tran'], '^-', label='Transformation', linewidth=2.5, markersize=4, alpha=0.9)

        plt.title(exp_name or 'MACAD Weight Evolution', fontsize=18, fontweight='bold')
        plt.xlabel('Training Step', fontsize=14)
        plt.ylabel('Weights', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(y_min, y_max)
        plt.tight_layout()

        filename = f"weight_realtime_{exp_name or 'macad'}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        import pickle

        history_filename = f"weight_history_{exp_name or 'macad'}.pkl"
        history_path = os.path.join(save_dir, history_filename)

        with open(history_path, 'wb') as f:
            pickle.dump(self.weight_step_history, f)

        print(f"Weight history (pickle) saved → {history_path}")
        print(f"\nReal-time weight evolution plot saved → {save_path}")
