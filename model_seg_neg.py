import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import backbone as encoder
from . import decoder
from torchvision.transforms import Compose, Normalize
import sys

import clip
from clip.clip_text import new_class_names, BACKGROUND_CATEGORY
from clip.clip_tool import generate_clip_fts



def Normalize_clip():
    return Compose([
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()


def _refine_cams(ref_mod, images, cams, valid_key):
    images = images.unsqueeze(0)
    cams = cams.unsqueeze(0)

    refined_cams = ref_mod(images.float(), cams.float())
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label.squeeze(0)


def cos_sim(x, y):
    x = F.normalize(x, p=2, dim=2, eps=1e-8)
    y = F.normalize(y, p=2, dim=2, eps=1e-8)
    cos_sim = torch.matmul(x, y.transpose(1, 2).contiguous())
    return torch.abs(cos_sim)


class Proj_head(nn.Module):
    def __init__(self, in_dim, out_dim=4096, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        # pdb.set_trace()
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            # trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, input, information):
        B, N1, C = input.shape
        B, N2, C = information.shape
        q = self.q(input).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(information).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(information).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


def cos_sim(x, y):
    x = F.normalize(x, p=2, dim=1, eps=1e-8)
    y = F.normalize(y, p=2, dim=1, eps=1e-8)
    cos_sim = torch.matmul(x, y.transpose(1,2).contiguous())
    return torch.abs(cos_sim)


class network(nn.Module):
    def __init__(self, backbone, num_classes=None, pretrained=None, init_momentum=None, aux_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.init_momentum = init_momentum

        self.encoder = getattr(encoder, backbone)(pretrained=pretrained, aux_layer=aux_layer)


        self.in_channels = [self.encoder.embed_dim] * 4 if hasattr(self.encoder, "embed_dim") else [self.encoder.embed_dims[-1]] * 4 

        self.pooling = F.adaptive_max_pool2d

        self.decoder = decoder.LargeFOV(in_planes=self.in_channels[-1], out_planes=self.num_classes)

        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes-1, kernel_size=1, bias=False,)
        self.aux_classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes-1, kernel_size=1, bias=False,)

        self.encoder_clip, _ = clip.load("./pretrained/ViT-B-16.pt", device="cuda:0")
        for name, param in self.encoder_clip.named_parameters():
            if "11" not in name:
                param.requires_grad = False
        for name, param in self.encoder_clip.named_parameters():
            print(name, param.requires_grad)

        self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'], self.encoder_clip)
        self.fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'], self.encoder_clip)

        self.proj_fg = Proj_head(in_dim=512, out_dim=self.in_channels[-1])
        self.proj_bg = Proj_head(in_dim=512, out_dim=self.in_channels[-1])
        for param, param_t in zip(self.proj_fg.parameters(), self.proj_bg.parameters()):
            param_t.data.copy_(param.data)  # initialize teacher with student
            param_t.requires_grad = False  # do not update by gradient

        self.cross_att_fg = CrossAttention(self.in_channels[-1])
        self.cross_att_bg = CrossAttention(self.in_channels[-1])
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.sigmoid = nn.Sigmoid()

    @torch.no_grad()
    def _EMA_update_encoder_teacher(self, n_iter=None):
        ## no scheduler here
        momentum = self.init_momentum
        for param, param_t in zip(self.proj_fg.parameters(), self.proj_bg.parameters()):
            param_t.data = momentum * param_t.data + (1. - momentum) * param.data

    def get_param_groups(self):

        param_groups = [[], [], [], []] # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.aux_classifier.weight)

        for param in list(self.proj_fg.parameters()):
            param_groups[2].append(param)
        for param in list(self.cross_att_fg.parameters()):
            param_groups[2].append(param)
        for param in list(self.cross_att_bg.parameters()):
            param_groups[2].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    def to_2D(self, x, h, w):
        n, hw, c = x.shape
        x = x.transpose(1, 2).contiguous().reshape(n, c, h, w)
        return x

    def get_pred(self, prototype, feature):
        pro_pred = F.relu(cos_sim(prototype, feature))

        return pro_pred

    def forward(self, x, cls_label=None, cam_only=False, istrain=False, n_iter=None, bg_img=None):

        self._EMA_update_encoder_teacher(n_iter)

        cls_token, _x, x_aux = self.encoder.forward_features(x)
        if istrain:
            b_bg, n_bg, c_bg, h_bg, w_bg = bg_img.shape
            bg_img = bg_img.view(b_bg*n_bg, c_bg, h_bg, w_bg)
            bg_token, _, _ = self.encoder.forward_features(bg_img)
            bg_token = bg_token.reshape(b_bg, n_bg, -1)

        h, w = x.shape[-2] // self.encoder.patch_size, x.shape[-1] // self.encoder.patch_size

        _x4 = self.to_2D(_x, h, w)
        _x_aux = self.to_2D(x_aux, h, w)



        if cam_only:

            cam = F.conv2d(_x4, self.classifier.weight).detach()
            cam_aux = F.conv2d(_x_aux, self.aux_classifier.weight).detach()

            return cam_aux, cam
        ################## clip ################
        #fts_all, attn_weight_list = generate_clip_fts(x, self.encoder_clip, require_all_fts=True)
        cls_aux = self.pooling(_x_aux, (1, 1))
        cls_aux = self.aux_classifier(cls_aux)

        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)

        cls_x4 = cls_x4.view(-1, self.num_classes - 1)
        cls_aux = cls_aux.view(-1, self.num_classes - 1)

        cls_p_label = torch.zeros_like(cls_x4)
        cls_p_label[cls_x4> -3] = 1

        img_feat = x_aux

        fg_tfeat = self.fg_text_features.unsqueeze(0).float()
        bg_tfeat = self.bg_text_features.unsqueeze(0).float()
        fg_tfeat = self.proj_fg(fg_tfeat)
        bg_tfeat = self.proj_bg(bg_tfeat)
        b = img_feat.size(0)
        fg_tfeat = fg_tfeat.repeat(b, 1, 1)
        bg_tfeat = bg_tfeat.repeat(b, 1, 1)
        cls_p_label = cls_p_label.unsqueeze(2)
        fg_tfeat = self.cross_att_fg(fg_tfeat, img_feat)*cls_p_label
        bg_tfeat = self.cross_att_bg(bg_tfeat, img_feat)

        bg_tfeat = self.max_pool(bg_tfeat.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        mix_feat = torch.cat((bg_tfeat, fg_tfeat), dim=1)  #[4,21,768]

        pro_pred = self.get_pred(mix_feat, img_feat)

        pro_pred = pro_pred.view(b, -1, h, w)
        pro_pred = F.interpolate(pro_pred, size=x.shape[2:], mode='bilinear', align_corners=True)

        ###
        pro_pred_feat = self.get_pred(mix_feat, _x).view(b, -1, h, w)[:,1:,...]

        seg = self.decoder(_x4, pro_pred_feat)


        
        if not istrain:
            return cls_x4, seg, _x4, cls_aux
        else:
            return cls_x4, seg, _x4, cls_aux, pro_pred, mix_feat, cls_token, bg_token