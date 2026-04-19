import torch
import torch as t
from torch import nn
import torch.nn.functional as F
import pickle
import numpy as np
import math
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class LightGCN_promax(BaseModel):
    def __init__(self, data_handler):
        super(LightGCN_promax, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.keep_rate = configs['model']['keep_rate']
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.sdr_weight = self.hyper_config['sdr_weight']
        self.s2dr_weight = self.hyper_config['s2dr_weight']

        self.usrprf_embeds = t.tensor(self.pca(configs['usrprf_embeds'])).float().cuda()
        self.itmprf_embeds = t.tensor(self.pca(configs['itmprf_embeds'])).float().cuda()  # [item_num, 1536]

        with open("../data/{}/new_trn_rag_mat.pkl".format(configs['data']['name']), 'rb') as fs:
            mat = (pickle.load(fs) != 0).astype(np.float32)
        if type(mat) != coo_matrix:
            mat = coo_matrix(mat)

        self.new_adj = data_handler._make_torch_adj(mat)

    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)

    def forward(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]

        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]

        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)

        embeds = sum(embeds_list[1:])
        self.final_embeds = embeds

        return embeds[:self.user_num], embeds[self.user_num:]

    def get_InfoNCE_loss(self, embedding_1, embedding_2, temperature):
        embedding_1 = t.nn.functional.normalize(embedding_1, dim=1)
        embedding_2 = t.nn.functional.normalize(embedding_2, dim=1)

        pos_score = (embedding_1 * embedding_2).sum(dim=-1)
        pos_score = t.exp(pos_score / temperature)

        ttl_score = t.matmul(embedding_1, embedding_2.transpose(0, 1))
        ttl_score = t.exp(ttl_score / temperature).sum(dim=1)

        cl_loss = - t.log(pos_score / ttl_score + 10e-6)
        return t.mean(cl_loss)

    def get_InfoNCE_loss_cross(self, u1, i1, u2, i2, tau):
        u1 = torch.nn.functional.normalize(u1, dim=-1)
        u2 = torch.nn.functional.normalize(u2, dim=-1)
        i1 = torch.nn.functional.normalize(i1, dim=-1)
        i2 = torch.nn.functional.normalize(i2, dim=-1)

        logits_u1_i2 = (u1 @ i2.T) / tau  # [B, B]
        logits_u2_i1 = (u2 @ i1.T) / tau  # [B, B]
        labels = torch.arange(u1.size(0), device=u1.device)
        return 0.5 * (F.cross_entropy(logits_u1_i2, labels) + F.cross_entropy(logits_u2_i1, labels))

    def pca(self, embedding):
        mean = np.mean(embedding, axis=0)

        cov = np.cov(embedding - mean, rowvar=False)

        U, S, _ = np.linalg.svd(cov, full_matrices=False)

        projection_matrix = U[..., :self.embedding_size]

        Diagnals = np.sqrt(1 / S)[:self.embedding_size]
        projection_matrix = projection_matrix.dot(np.diag(Diagnals))

        final_emb = (embedding - mean).dot(projection_matrix)

        return final_emb

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)
        user_embeds_2, item_embeds_2 = self.forward(self.new_adj, self.keep_rate)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]

        uni_ancs = t.unique(ancs)
        uni_poss = t.unique(poss)

        user_ssl_loss = self.get_InfoNCE_loss(user_embeds[uni_ancs], user_embeds_2[uni_ancs], 0.2)
        item_ssl_loss = self.get_InfoNCE_loss(item_embeds[uni_poss], item_embeds_2[uni_poss], 0.2)

        inter_ssl_loss = self.get_InfoNCE_loss_cross(user_embeds[ancs], item_embeds[poss], user_embeds_2[ancs], item_embeds_2[poss], 0.2)

        with torch.no_grad():
            l2_usrprf_emb = torch.nn.functional.normalize(self.usrprf_embeds[uni_ancs], dim=1)  # [B * t]
            l2_itmprf_emb = torch.nn.functional.normalize(self.itmprf_embeds, dim=1)   # [N * t]

            similarity = l2_usrprf_emb @ l2_itmprf_emb.T / 0.2
            text_sims = torch.softmax(similarity, dim=1).detach()

            H = -(text_sims * (text_sims.clamp_min(1e-12).log())).sum(dim=-1)
            H_norm = H / math.log(text_sims.shape[-1] + 1e-12)

            w = (1.0 - H_norm).clamp(min=0.1, max=1.0)

        usr_norm = torch.nn.functional.normalize(user_embeds, dim=1)  # [M * d]
        itm_norm = torch.nn.functional.normalize(item_embeds, dim=1)  # [N * d]

        inter_logits = (usr_norm[uni_ancs] @ itm_norm.T) / 0.2
        inter_sims = torch.softmax(inter_logits, dim=1)
        ce = - (text_sims * torch.log(inter_sims + 1e-8)).sum(dim=-1)

        sdr_loss = self.sdr_weight * (w * ce).mean()
        s2dr_loss = self.s2dr_weight * (user_ssl_loss + item_ssl_loss + inter_ssl_loss)

        bpr_loss = (cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0])
        reg_loss = self.reg_weight * reg_params(self)

        loss = bpr_loss + reg_loss + sdr_loss + s2dr_loss

        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'sdr_loss': sdr_loss,
                  's2dr_loss:': s2dr_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        pck_user_embeds = user_embeds[pck_users]

        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
