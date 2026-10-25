import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score, \
    precision_recall_curve, precision_score, recall_score
import warnings
import matplotlib


warnings.filterwarnings("ignore")  # 모든 경고 무시
matplotlib.set_loglevel("error")   # matplotlib 로그 최소화
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.ERROR)  # PNG 관련 로그 차단

def get_logger(log_file):
    # A dedicated logger prevents handlers from a previous seed run writing
    # into the next seed's file when several seeds execute in one process.
    logger = logging.getLogger(os.path.abspath(log_file))
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    def log(s):
        logger.info(s)

    return log


# We keep the dictionary of metrics for each dataset and task
dict_metrics = {"smd": {"acc": accuracy_score, "mac_f1": f1_score, "w_f1": f1_score},
                "msl": {"acc": accuracy_score, "mac_f1": f1_score, "w_f1": f1_score},
                "boiler": {"acc": accuracy_score, "mac_f1": f1_score, "w_f1": f1_score},
                "smap": {"acc": accuracy_score, "mac_f1": f1_score, "w_f1": f1_score},
                }


def get_dataset_type(args):
    path = str(args.path_src).lower()
    if "smd" in path:
        return "smd"
    elif "msl" in path:
        return "msl"
    elif "smap" in path:
        return "smap"
    elif "boiler" in path:
        return "boiler"
    else:
        return "sensor"

# 특정 메트릭의 현재 값과 평균값을 추적
# L_Src_Sup 1.9993e+00 (1.9993e+00) L_Trg_Inj 1.9997e+00 (1.9997e+00) Loss Disc 7.1272e-01 (7.1272e-01)
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# 훈련/검증 진행 상황을 시각화하고 로깅
# Epoch: [0][9/9]	L_Src_Sup 1.9993e+00 (1.9993e+00)	L_Trg_Inj 1.9997e+00 (1.9997e+00)	Loss Disc 7.1272e-01 (7.1272e-01)	Loss Pred 7.7916e-01 (7.8367e-01)	Loss TOTAL 1.5354e+00 (1.5399e+00)
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters 
        self.prefix = prefix

    def display(self, batch, is_logged=False):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # If the progress will be logged (instead of printing)
        # we will return the string so that it can be logged in the main script.
        if not is_logged:
            print('\t'.join(entries))
        else:
            return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches): # [001/100] 배치 진행 표시 형식 생성
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

# 모델의 예측 결과를 저장하고 이상 탐지 메트릭 계산
class PredictionMeter(object):
    def __init__(self, args):
        self.args = args
        self.dataset_type = get_dataset_type(args)
        self.target_list = []
        self.output_list = []

        # Initialize the metric dictionary to be returned
        self.dict_metrics = {}
        self.dict_metrics = dict_metrics[self.dataset_type]

    def update(self, target, output):
        if self.dataset_type == "smd":
            output_np = output.detach().cpu().numpy().flatten()
        elif self.dataset_type == "msl":
            output_np = output.detach().cpu().numpy().flatten()
        elif self.dataset_type == "boiler":
            output_np = output.detach().cpu().numpy().flatten()
        elif self.dataset_type == "smap":
            output_np = output.detach().cpu().numpy().flatten()
        else:
            output_np = output.detach().cpu().numpy().argmax(axis=1).flatten()
        target_np = target.detach().cpu().numpy().flatten()

        self.output_list = self.output_list + list(output_np)
        self.target_list = self.target_list + list(target_np)

    def get_metrics(self):
        return_dict = {}
        output = np.array(self.output_list)
        target = np.array(self.target_list)

        avg_prc = average_precision_score(target, output, pos_label=1)
        roc_auc =0
        try:
            roc_auc = roc_auc_score(target, output)
        except ValueError:
            pass
        prec, rec, thr = precision_recall_curve(target, output, pos_label=1)
        # nan 값 대체
        prec = np.where(prec == np.nan, 0.0, prec)
        rec = np.where(rec == np.nan, 0.0, rec)

        with np.errstate(invalid='ignore'):
            f1score = np.where((rec + prec) == 0.0, 0.0, 2 * prec * rec / (prec + rec))
        best_f1_index = np.argmax(f1score)

        # 최대 f1 score를 갖는 값들을 저장
        return_dict["best_f1"] = f1score[best_f1_index]
        return_dict["best_prec"] = prec[best_f1_index]
        return_dict["best_rec"] = rec[best_f1_index]
        return_dict["best_thr"] = thr[best_f1_index]
        return_dict["avg_prc"] = avg_prc
        return_dict["roc_auc"] = roc_auc
        output = np.where(output[:] > thr[best_f1_index], 1, 0) # 최적 임계값
        return_dict["macro_F1"] = f1_score(target, output, average="macro")
        return return_dict

    def get_metrics_at_threshold(self, threshold):
        """Evaluate a threshold fixed without access to this meter's labels."""
        output = np.asarray(self.output_list)
        target = np.asarray(self.target_list)
        predictions = (output > threshold).astype(np.int64)
        try:
            roc_auc = roc_auc_score(target, output)
        except ValueError:
            roc_auc = 0.0
        return {
            "avg_prc": average_precision_score(target, output, pos_label=1),
            "roc_auc": roc_auc,
            "f1": f1_score(target, predictions, pos_label=1, zero_division=0),
            "precision": precision_score(target, predictions, pos_label=1, zero_division=0),
            "recall": recall_score(target, predictions, pos_label=1, zero_division=0),
            "threshold": float(threshold),
        }


