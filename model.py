import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn_layers_1(nn.Module):
    """
    CNN layers applied on acc sensor data to generate pre-softmax
    ---
    params for __init__():
        input_size: e.g. 1
        num_classes: e.g. 6
    forward():
        Input: data
        Output: pre-softmax
    """

    def __init__(self, input_size):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_size, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Conv2d(64, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Conv2d(32, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(3, 1)),  # 3
        )  # torch.Size([1, 16, 97, 4]) 6208

    def forward(self, x):
        x = self.features(x)  # [bsz, 16, 1, 198]
        return x


class cnn_layers_2(nn.Module):
    """
    CNN layers applied on video data to generate pre-softmax
    ---
    params for __init__():
        input_size: e.g. 1
        num_classes: e.g. 6
    forward():
        Input: data
        Output: pre-softmax
    """

    def __init__(self, input_size):
        super().__init__()

        # Extract features, 3D conv layers
        self.features = nn.Sequential(
            nn.Conv3d(input_size, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(3),
        )

    def forward(self, x):

        x = self.features(x)

        return x


class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.imu_cnn_layers = cnn_layers_1(input_size)
        self.image_cnn_layers = cnn_layers_2(input_size * 3)

    def forward(self, x1, x2):

        imu_output = self.imu_cnn_layers(x1)
        image_output = self.image_cnn_layers(x2)

        return imu_output, image_output


def get_encoded_feature_dim(dataset):
    if dataset == "UTD-MHAD":
        return 3648, 12288
    elif dataset == "WEAR":
        return 6208, 10240
    else:  # ours
        return 6208, 7680
    # raise RuntimeError("dataset name error")


class FeatureExtractor(nn.Module):
    """Feature extractor for human-activity-recognition."""

    def __init__(self, input_size, dataset):
        super().__init__()

        self.encoder = Encoder(input_size)
        dim_h1, dim_h2 = get_encoded_feature_dim(dataset)

        self.head_1 = nn.Sequential(
            nn.Linear(dim_h1, 512),  # 6208
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
        )

        self.head_2 = nn.Sequential(
            nn.Linear(dim_h2, 512),  # 7680
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
        )

    def forward(self, x1, x2):
        imu_output, video_output = self.encoder(x1, x2)

        imu_output = F.normalize(
            self.head_1(imu_output.view(imu_output.size(0), -1)), dim=1
        )
        video_output = F.normalize(
            self.head_2(video_output.view(video_output.size(0), -1)), dim=1
        )

        return imu_output, video_output


## contrastive fusion loss with SupCon format: https://arxiv.org/pdf/2004.11362.pdf
class ConFusionLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(ConFusionLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda:1')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        # os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        mask = torch.eye(batch_size, dtype=torch.float32).cuda()

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(
            torch.unbind(features, dim=1), dim=0
        )  # change to [n_views*bsz, 3168]
        contrast_feature = F.normalize(contrast_feature, dim=1)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits, z_i * z_a / T
        similarity_matrix = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  # positive index
        # print(mask.shape)#[1151, 1152] (btz*9)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0,
        )
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        #     0
        # )#dig to 0, others to 1 (negative samples)

        mask = mask * logits_mask  # positive samples except itself

        # compute log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask  # exp(z_i * z_a / T)

        # SupCon out
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # sup_out

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class Attn(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        dim_h1, dim_h2 = get_encoded_feature_dim(dataset)
        self.reduce_d1 = nn.Sequential(nn.Linear(dim_h1, 512))
        self.reduce_d2 = nn.Sequential(nn.Linear(dim_h2, 512))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=2, dim_feedforward=64
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, hidden_state_1, hidden_state_2):
        new_1 = self.reduce_d1(hidden_state_1.view(hidden_state_1.size(0), -1))
        new_2 = self.reduce_d2(hidden_state_2.view(hidden_state_2.size(0), -1))

        h = torch.stack((new_1, new_2), dim=0)  # [2, bsz, 1024]
        h = self.transformer_encoder(h)
        return h


class LinearClassifierAttn(nn.Module):
    """Linear classifier"""

    def __init__(self, num_classes, dataset):
        super(LinearClassifierAttn, self).__init__()

        self.attn = Attn(dataset)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, feature1, feature2):
        fused_feature = self.attn(feature1, feature2)
        fused_feature = fused_feature.permute(1, 0, 2)
        fused_feature = fused_feature.contiguous().view(fused_feature.size(0), -1)
        output = self.classifier(fused_feature)
        return output
