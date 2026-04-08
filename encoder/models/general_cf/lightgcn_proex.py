import torch
import torch as t
from torch import nn
import torch.nn.functional as F
import pickle
import numpy as np
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class LightGCN_proex(BaseModel):
    def __init__(self, data_handler):
        super(LightGCN_proex, self).__init__(data_handler)
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
        self.num_profiles = configs['model']['profiles']
        self.num_envs = self.hyper_config['num_envs']
        self.inv_weight = self.hyper_config['inv_weight']
        self.ex_weight = self.hyper_config['ex_weight']

        self.alpha = torch.Tensor([self.hyper_config['alpha']] * self.num_profiles)

        self.usrprf_embeds = []
        self.itmprf_embeds = []

        self.usrprf_embeds.append(t.tensor(configs['usrprf_embeds']).float().cuda())
        self.itmprf_embeds.append(t.tensor(configs['itmprf_embeds']).float().cuda())  # [item_num, 1536]

        for i in range(self.num_profiles - 1):
            usrprf_embeds_path = "../data/{}/usr_emb_np_{}.pkl".format(configs['data']['name'], i + 1)
            itmprf_embeds_path = "../data/{}/itm_emb_np_{}.pkl".format(configs['data']['name'], i + 1)
            with open(usrprf_embeds_path, 'rb') as f:
                self.usrprf_embeds.append(t.tensor(pickle.load(f)).float().cuda())
            with open(itmprf_embeds_path, 'rb') as f:
                self.itmprf_embeds.append(t.tensor(pickle.load(f)).float().cuda())

        self.mlp = nn.Sequential(
            nn.Linear(self.usrprf_embeds[0].shape[1], (self.usrprf_embeds[0].shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrprf_embeds[0].shape[1] + self.embedding_size) // 2, self.embedding_size)
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

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
        embedding_1 = t.nn.functional.normalize(embedding_1)
        embedding_2 = t.nn.functional.normalize(embedding_2)

        pos_score = (embedding_1 * embedding_2).sum(dim=-1)
        pos_score = t.exp(pos_score / temperature)

        ttl_score = t.matmul(embedding_1, embedding_2.transpose(0, 1))
        ttl_score = t.exp(ttl_score / temperature).sum(dim=1)

        cl_loss = - t.log(pos_score / ttl_score + 10e-6)
        return t.mean(cl_loss)

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]

        env_losses = []
        usr_prf_list = []
        pos_prf_list = []
        neg_prf_list = []

        uni_ancs = t.unique(ancs)
        uni_poss = t.unique(poss)

        uniform_loss = 0.

        for i in range(self.num_profiles):
            all_usrprf_embeds = self.mlp(self.usrprf_embeds[i])
            all_itmprf_embeds = self.mlp(self.itmprf_embeds[i])

            ancprf_embeds = all_usrprf_embeds[ancs]
            posprf_embeds = all_itmprf_embeds[poss]
            negprf_embeds = all_itmprf_embeds[negs]

            uniform_loss += self.get_InfoNCE_loss(all_usrprf_embeds[uni_ancs], all_usrprf_embeds[uni_ancs], 0.2)
            uniform_loss += self.get_InfoNCE_loss(all_itmprf_embeds[uni_poss], all_itmprf_embeds[uni_poss], 0.2)

            usr_prf_list.append(ancprf_embeds)
            pos_prf_list.append(posprf_embeds)
            neg_prf_list.append(negprf_embeds)

        ancprf_embeddings = t.stack(usr_prf_list)
        posprf_embeddings = t.stack(pos_prf_list)
        negprf_embeddings = t.stack(neg_prf_list)

        for env in range(self.num_envs):
            beta = t.distributions.Dirichlet(self.alpha).sample().cuda()

            ancprf_embeds = t.einsum('i,ibd->bd', beta, ancprf_embeddings)
            posprf_embeds = t.einsum('i,ibd->bd', beta, posprf_embeddings)
            negprf_embeds = t.einsum('i,ibd->bd', beta, negprf_embeddings)

            user_emb = anc_embeds + ancprf_embeds
            pos_emb = pos_embeds + posprf_embeds
            neg_emb = neg_embeds + negprf_embeds

            bpr_loss = (cal_bpr_loss(user_emb, pos_emb, neg_emb) / anc_embeds.shape[0])

            # The alignment loss within the environment is disabled by default,
            # as this process provides limited performance improvement but significantly increases complexity.
            # align_loss = 0.
            # align_loss += self.get_InfoNCE_loss(anc_embeds, ancprf_embeds, 0.2)
            # align_loss += self.get_InfoNCE_loss(pos_embeds, posprf_embeds, 0.2)
            # align_loss += self.get_InfoNCE_loss(neg_embeds, negprf_embeds, 0.2)
            #
            # env_loss = bpr_loss + self.align_weight * align_loss

            env_losses.append(bpr_loss)

        reg_loss = self.reg_weight * reg_params(self)

        invariant_loss = t.stack(env_losses).var()
        main_loss = t.stack(env_losses).sum()

        loss = main_loss + reg_loss + self.inv_weight * invariant_loss + self.ex_weight * uniform_loss

        losses = {'bpr_loss': main_loss, 'reg_loss': reg_loss, 'inv_loss': invariant_loss, 'uniform_loss:': uniform_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
        self.is_training = False

        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        pck_user_embeds = user_embeds[pck_users]

        usr_prf_list = []
        itm_prf_list = []

        for env in range(self.num_profiles):
            all_usrprf_embeds = self.mlp(self.usrprf_embeds[env])
            all_itmprf_embeds = self.mlp(self.itmprf_embeds[env])

            ancprf_embeds = all_usrprf_embeds[pck_users]
            posprf_embeds = all_itmprf_embeds

            usr_prf_list.append(ancprf_embeds)
            itm_prf_list.append(posprf_embeds)

        ancprf_embeddings = t.stack(usr_prf_list, dim=1)
        posprf_embeddings = t.stack(itm_prf_list, dim=1)

        # mean pooling for test
        user_emb = pck_user_embeds + t.mean(ancprf_embeddings, dim=1)
        item_emb = item_embeds + t.mean(posprf_embeddings, dim=1)

        full_preds = user_emb @ item_emb.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
