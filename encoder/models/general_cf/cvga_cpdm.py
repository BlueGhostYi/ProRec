import torch
from torch import nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params
from models.base_model import BaseModel

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class cvga_CPDM(BaseModel):
    def __init__(self, data_handler):
        super(cvga_CPDM, self).__init__(data_handler)

        self.beta = self.hyper_config['beta']
        self.embedding_size = configs['model']['embedding_size']

        self.data_handler = data_handler

        self.p_dims = [self.embedding_size, self.item_num]
        self.q_dims = [self.item_num, self.embedding_size]
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]

        self.q_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])]
        )

        self.p_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])]
        )

        self.drop = nn.Dropout(self.hyper_config['dropout'])

        self.usrprf_embeds = torch.tensor(configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds = torch.tensor(configs['itmprf_embeds']).float().cuda()  # [item_num, 1536]

        self.mlp = nn.Sequential(
            nn.Linear(self.itmprf_embeds.shape[1], 600),
            nn.Tanh(),
            nn.Linear(600, self.embedding_size * 2)
        )

        self.Graph = self.sparse_adjacency_matrix_R()
        self.Graph = self.convert_sp_mat_to_sp_tensor(self.Graph)  # sparse tensor
        self.Graph = self.Graph.coalesce().to(configs['device'])  # Sort the edge index and remove redundancy

        self.final_embeds = None
        self.is_training = False

    def convert_sp_mat_to_sp_tensor(self, sp_mat):
        """
            coo.row: x in user-item graph
            coo.col: y in user-item graph
            coo.data: [value(x,y)]
        """
        coo = sp_mat.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        value = torch.FloatTensor(coo.data)
        # from a sparse matrix to a sparse float tensor
        sp_tensor = torch.sparse.FloatTensor(index, value, torch.Size(coo.shape))
        return sp_tensor

    def sparse_adjacency_matrix_R(self):
        try:
            norm_adjacency = sp.load_npz("../data/" + configs['data']['name'] + '/pre_R.npz')
            print("\t Adjacency matrix loading completed.")
        except:
            adjacency_matrix = self.data_handler.train_data

            row_sum = np.array(adjacency_matrix.sum(axis=1))
            row_d_inv = np.power(row_sum, -0.5).flatten()
            row_d_inv[np.isinf(row_d_inv)] = 0.
            row_degree_matrix = sp.diags(row_d_inv)

            col_sum = np.array(adjacency_matrix.sum(axis=0))
            col_d_inv = np.power(col_sum, -0.5).flatten()
            col_d_inv[np.isinf(col_d_inv)] = 0.
            col_degree_matrix = sp.diags(col_d_inv)

            norm_adjacency = row_degree_matrix.dot(adjacency_matrix).dot(col_degree_matrix).tocsr()
            sp.save_npz("../data/" + configs['data']['name'] + '/pre_R.npz', norm_adjacency)
            print("\t Adjacency matrix constructed.")

        return norm_adjacency

    def encode(self):

        for i, layer in enumerate(self.q_layers):
            if i == 0:
                h = layer(self.Graph)
            else:
                h = layer(torch.sparse.mm(self.Graph.t(), h))
            h = self.drop(h)

            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]

        return mu, logvar

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def reparameterize(self, mu, logvar):
        if self.is_training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std) + mu
        else:
            z = mu
        return z

    def cal_loss(self, user, batch_data):
        self.is_training = True
        mu_src, logvar_src = self.encode()

        user_emb = self.usrprf_embeds[user]

        h = self.drop(batch_data)
        # [batch, item_num] * [item_num, dim] = [batch, dim]
        hidden = torch.matmul(h, self.itmprf_embeds) + user_emb
        hidden = self.mlp(hidden)

        mu_llm = hidden[:, :self.embedding_size]
        logvar_llm = hidden[:, self.embedding_size:]

        # There are two processing strategies:
        # (1) performing downstream tasks after addition, and
        # (2) directly conducting downstream tasks.
        # In practice, we found that these two strategies have their own advantages on different datasets.
        # Therefore, we adopt a unified design that performs reconstruction and matching after addition.
        mu = mu_src[user] + mu_llm
        logvar = logvar_src[user] + logvar_llm

        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        BCE = - torch.mean(torch.sum(F.log_softmax(recon_x, 1) * batch_data, -1))
        KLD = - 0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        mu_mix = (mu + mu_llm) / 2
        logvar_mix = (logvar + logvar_llm) / 2

        KLD_1 = - 0.5 * torch.mean(torch.sum(1 + torch.log(logvar.exp()/(logvar_mix.exp() + 10e-8) + 10e-8) -
                                           (mu - mu_mix).pow(2)/(logvar_mix.exp() + 10e-8) - logvar.exp()/(logvar_mix.exp() + 10e-8), dim=1))

        KLD_2 = - 0.5 * torch.mean(torch.sum(1 + torch.log(logvar_llm.exp() / (logvar_mix.exp() + 10e-8) + 10e-8) -
                                             (mu_llm - mu_mix).pow(2) / (logvar_mix.exp() + 10e-8) - logvar_llm.exp() / (
                                                         logvar_mix.exp() + 10e-8), dim=1))

        KLD = KLD + self.beta * (KLD_1 + KLD_2)

        loss = BCE + KLD
        losses = {'bpr_loss': BCE, 'reg_loss': KLD}
        return loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        mu, logvar = self.encode()

        data = self.data_handler.train_data[pck_users.cpu()]
        data = torch.FloatTensor(data.toarray()).to(configs['device'])

        user_emb = self.usrprf_embeds[pck_users]

        # [batch, item_num] * [item_num, dim] = [batch, dim]
        hidden = torch.matmul(data, self.itmprf_embeds) + user_emb

        hidden = self.mlp(hidden)

        mu_llm = hidden[:, :self.embedding_size]
        logvar_llm = hidden[:, self.embedding_size:]

        mu = mu[pck_users] + mu_llm
        logvar = logvar[pck_users] + logvar_llm

        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        full_preds = self._mask_predict(recon_x, train_mask)
        return full_preds