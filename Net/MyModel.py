import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
from .Embedding import My_Entity_Aware_Embedding
from .Encoder import MyPCNN_V


class MyHierRelLayer(nn.Module):
    def __init__(self, hier_rel_num, hier_rel_dim, linear_in, linear_out, mlp_hidden_size, mlp_out, mlp_bias=False):
        super(MyHierRelLayer, self).__init__()
        self.hier_rel_matrix = Parameter(torch.empty(size=[hier_rel_num, hier_rel_dim], requires_grad=True))
        self.hier_rel_gate = nn.Linear(linear_in, linear_out)
        self.hier_mlp_1 = nn.Linear(linear_out, mlp_hidden_size, bias=mlp_bias)
        self.hier_mlp_2 = nn.Linear(mlp_hidden_size, mlp_out, bias=mlp_bias)
        self.layer_norm = nn.LayerNorm(mlp_out)
        self.mlp_bias = mlp_bias
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.hier_rel_matrix)
        nn.init.xavier_uniform_(self.hier_rel_gate.weight) 
        nn.init.xavier_uniform_(self.hier_mlp_1.weight)
        nn.init.xavier_uniform_(self.hier_mlp_2.weight)
        nn.init.zeros_(self.hier_rel_gate.bias)
        if self.mlp_bias:
            nn.init.zeros_(self.hier_mlp_1.bias)
            nn.init.zeros_(self.hier_mlp_2.bias)

    def logit_select(self, hier_logit, X_Scope):
        tower_repre = []
        for s in X_Scope:
            one_bag, _ = torch.max(hier_logit[s[0]:s[1]], dim=0)
            tower_repre.append(one_bag)
        stack_repre = torch.stack(tower_repre, dim=0)
        return stack_repre

    def forward(self, S):  
        # S [bs, h*3]
        "hier_rel"
        hier_logits = torch.matmul(S, self.hier_rel_matrix.t())
        hier_index = F.softmax(hier_logits, dim=-1)
        hier_relation = torch.matmul(hier_index, self.hier_rel_matrix)  # [bs, h*3],  relation-aware repre
        "gate"
        concat_hier = torch.cat([S, hier_relation], dim=-1)
        alpha_hier = torch.sigmoid(self.hier_rel_gate(concat_hier))  # gate
        context_hier = alpha_hier * S + (1 - alpha_hier) * hier_relation  # relation-agument repre  , [bs, h*3]
        "MLP linear"
        middle_hier = F.relu(self.hier_mlp_1(context_hier))
        output_hier = self.hier_mlp_2(middle_hier)  # [bs, h*3]
        "add&norm"
        output_hier += S  # [bs, h*3]
        output_hier = self.layer_norm(output_hier)
        return hier_logits, output_hier

class MyHierRelLayer_V(nn.Module):
    def __init__(self, hier_rel_num, hier_rel_dim, linear_in, linear_out, mlp_hidden_size, mlp_out, mlp_bias=False):
        super(MyHierRelLayer_V, self).__init__()
        self.hier_rel_matrix = Parameter(torch.empty(size=[hier_rel_num, hier_rel_dim], requires_grad=True))
        self.hier_rel_gate = nn.Linear(linear_in, linear_out)
        self.hier_update_gate = nn.Linear(linear_in, linear_out)
        self.S_update_gate = nn.Linear(linear_in, linear_out)
        self.hier_mlp_1 = nn.Linear(linear_out, mlp_hidden_size, bias=mlp_bias)
        self.hier_mlp_2 = nn.Linear(mlp_hidden_size, mlp_out, bias=mlp_bias)
        self.layer_norm = nn.LayerNorm(mlp_out)
        self.mlp_bias = mlp_bias
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.hier_rel_matrix)
        nn.init.xavier_uniform_(self.hier_rel_gate.weight) 
        nn.init.xavier_uniform_(self.hier_update_gate.weight)
        nn.init.xavier_uniform_(self.S_update_gate.weight)
        nn.init.xavier_uniform_(self.hier_mlp_1.weight)
        nn.init.xavier_uniform_(self.hier_mlp_2.weight)
        nn.init.zeros_(self.hier_rel_gate.bias)
        nn.init.zeros_(self.hier_update_gate.bias)
        nn.init.zeros_(self.S_update_gate.bias)
        if self.mlp_bias:
            nn.init.zeros_(self.hier_mlp_1.bias)
            nn.init.zeros_(self.hier_mlp_2.bias)    

    def forward(self, hier, S):
        # S [bs, h*3]
        "sent2rel"
        logit_t_hier = torch.matmul(hier, self.hier_rel_matrix.t())
        logit = torch.matmul(S, self.hier_rel_matrix.t())
        logit_t_hier_pro = F.softmax(logit_t_hier, dim=-1)
        logit_t_S_pro = F.softmax(logit, dim=-1)
        c_t_hier = torch.matmul(logit_t_hier_pro, self.hier_rel_matrix)
        c_t_S = torch.matmul(logit_t_S_pro, self.hier_rel_matrix)
        concat_hier = torch.cat([hier, S], dim=-1)
        alpha_hier = torch.sigmoid(self.hier_rel_gate(concat_hier))  # gate
        relation_t = alpha_hier * c_t_hier + (1 - alpha_hier) * c_t_S
        "update hier"
        concat_update_hier = torch.cat([hier, relation_t], dim=-1)
        alpha_update_hier = torch.sigmoid(self.hier_update_gate(concat_update_hier))  # gate
        hier_t = alpha_update_hier * hier + (1 - alpha_update_hier) * relation_t
        "update S"
        concat_update_S = torch.cat([S, relation_t], dim=-1)
        alpha_update_S = torch.sigmoid(self.S_update_gate(concat_update_S))  # gate
        update_S = alpha_update_S * S + (1 - alpha_update_S) * relation_t
        "MLP linear"
        middle_hier = F.relu(self.hier_mlp_1(update_S))
        output_hier = self.hier_mlp_2(middle_hier)  # [bs, h*3]
        "add&norm"
        output_hier += S  # [bs, h*3]
        output_hier = self.layer_norm(output_hier)
        return logit_t_hier, logit, hier_t, output_hier


class MyDenseNet(nn.Module):
    def __init__(self, hier1_rel_num, hier2_rel_num, hier3_rel_num, hier_rel_dim, linear_in, linear_out, mlp_hidden_size, mlp_out, mlp_bias=False, hier_rel_net=None):
        super(MyDenseNet, self).__init__()
        self.hier_rel_net = hier_rel_net
        if self.hier_rel_net == "rhia":
            self.root_rel = Parameter(torch.empty(size=[1, hier_rel_dim], requires_grad=True))
            self.hier1_rel_net = MyHierRelLayer_V(hier1_rel_num, hier_rel_dim, linear_in, linear_out, mlp_hidden_size, mlp_out, mlp_bias=False)
            self.hier2_rel_net = MyHierRelLayer_V(hier2_rel_num, hier_rel_dim, linear_in, linear_out, mlp_hidden_size, mlp_out, mlp_bias=False)
            self.hier3_rel_net = MyHierRelLayer_V(hier3_rel_num, hier_rel_dim, linear_in, linear_out, mlp_hidden_size, mlp_out, mlp_bias=False)
        else:  ## CoRA
            self.hier1_rel_net = MyHierRelLayer(hier1_rel_num, hier_rel_dim, linear_in, linear_out, mlp_hidden_size, mlp_out, mlp_bias=False)
            self.hier2_rel_net = MyHierRelLayer(hier2_rel_num, hier_rel_dim, linear_in, linear_out, mlp_hidden_size, mlp_out, mlp_bias=False)
            self.hier3_rel_net = MyHierRelLayer(hier3_rel_num, hier_rel_dim, linear_in, linear_out, mlp_hidden_size, mlp_out, mlp_bias=False)
        self.init_weight()

    def init_weight(self):
        if self.hier_rel_net == "rhia":
            nn.init.zeros_(self.root_rel)
        else:
            print("Defaults Net has built.")

    def forward(self, S):
        if self.hier_rel_net == "rhia":
            # hier1 
            root = self.root_rel.expand(S.shape)
            hier1_logits_hier, hier1_logits_S, hier1, output_hier1 = self.hier1_rel_net(root, S)
            # hier2
            hier2_logits_hier, hier2_logits_S, hier2, output_hier2 = self.hier2_rel_net(hier1, S)
            # hier3
            hier3_logits_hier, hier3_logits_S, hier3, output_hier3 = self.hier3_rel_net(hier2, S)
            return hier1_logits_hier, hier1_logits_S, output_hier1, hier2_logits_hier, hier2_logits_S, output_hier2, hier3_logits_hier, hier3_logits_S, output_hier3, hier3
        else:
            hier1_logits, output_hier1 = self.hier1_rel_net(S)
            hier2_logits, output_hier2 = self.hier2_rel_net(S)
            hier3_logits, output_hier3 = self.hier3_rel_net(S)
            return hier1_logits, output_hier1, hier2_logits, output_hier2, hier3_logits, output_hier3


class MyModel(nn.Module):
    def __init__(self, pre_word_vec, hier1_rel_num, hier2_rel_num, hier3_rel_num, lambda_pcnn=0.05, pos_dim=5, pos_len=100, hidden_size=230, dropout_rate=0.5, embed_type=None, ent_order=None, hier_rel_net=None):
        super(MyModel, self).__init__()
        word_embedding = torch.from_numpy(np.load(pre_word_vec))
        word_dim = word_embedding.shape[-1]
        self.hidden_size = hidden_size
        self.ent_order = ent_order
        self.hier_rel_net = hier_rel_net
        # embedding
        self.embedding = My_Entity_Aware_Embedding(word_embedding, word_dim, lam=lambda_pcnn, pos_dim=pos_dim, pos_len=pos_len)
        # encoder
        input_embed_dim = 3 * word_dim
        self.PCNN = MyPCNN_V(input_embed_dim, lambda_pcnn, hidden_size)
        # hier_rel augment
        self.dense_net = MyDenseNet(hier1_rel_num, hier2_rel_num, hier3_rel_num, 3 * hidden_size, 2 * 3 * hidden_size, 3 * hidden_size, 1024, 3 * hidden_size, mlp_bias=False, hier_rel_net=self.hier_rel_net)
        # selector
        combine_feature_dim = 3 * 3 * hidden_size
        if self.hier_rel_net == "rhia":
            self.bag_att_layer = nn.Linear(3 * hidden_size * 2, 1, bias=False)
        else:
            self.bag_att_layer = nn.Linear(combine_feature_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        # classifier
        self.classifer = nn.Linear(combine_feature_dim, hier3_rel_num)
        # classifier for entity order
        if self.ent_order == "eop":
            self.ent_order_logit = nn.Linear(3 * hidden_size * 3, 2)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.bag_att_layer.weight)
        nn.init.xavier_uniform_(self.classifer.weight)
        nn.init.zeros_(self.classifer.bias)
        if self.ent_order == "eop":
            nn.init.xavier_uniform_(self.ent_order_logit.weight)
            nn.init.zeros_(self.ent_order_logit.bias)
    
    def bag_att_V(self, output_hier, hier_info, X_Scope):
        prob_bag = self.bag_att_layer(hier_info)
        last_dim = output_hier.shape[-1]
        tower_repre = []
        for s in X_Scope:
            prob = F.softmax(torch.reshape(prob_bag[s[0]:s[1]], shape=(1, -1)), dim=1)
            one_bag = torch.reshape(torch.matmul(prob, output_hier[s[0]:s[1]]), shape=(last_dim, ))
            tower_repre.append(one_bag)
        stack_repre = torch.stack(tower_repre, dim=0)
        return stack_repre

    def bag_att(self, output_hier, X_Scope):
        prob_bag = self.bag_att_layer(output_hier)
        last_dim = output_hier.shape[-1]
        tower_repre = []
        for s in X_Scope:
            prob = F.softmax(torch.reshape(prob_bag[s[0]:s[1]], shape=(1, -1)), dim=1)
            one_bag = torch.reshape(torch.matmul(prob, output_hier[s[0]:s[1]]), shape=(last_dim, ))
            tower_repre.append(one_bag)
        stack_repre = torch.stack(tower_repre, dim=0)
        return stack_repre
    
    def forward(self, X, X_Pos1, X_Pos2, X_Order, Ent_Pos, X_Index1, X_Index2, X_Ent1, X_Ent2, X_Mask, X_Scope, X_length):
        # Embeding
        Xp, Xe, X = self.embedding(X, X_Pos1, X_Pos2, X_Ent1, X_Ent2)
        # Encoder
        S = self.PCNN(X, X_Mask)  # [?, 690]

        if self.hier_rel_net == "rhia":
            hier1_logits_hier, hier1_logits, output_hier1, hier2_logits_hier, hier2_logits, output_hier2, hier3_logits_hier, hier3_logits, output_hier3, hier_info = self.dense_net(S)
        else:
            hier1_logits, output_hier1, hier2_logits, output_hier2, hier3_logits, output_hier3 = self.dense_net(S)

        # combine features
        output_hier = torch.cat([output_hier1, output_hier2, output_hier3], dim=1)

        # Selector
        if self.hier_rel_net == "rhia":
            hier_info = torch.cat([S, hier_info], dim=-1)
            X = self.bag_att_V(output_hier, hier_info, X_Scope)
        else:
            X = self.bag_att(output_hier, X_Scope)
        
        # Classifier
        X = self.dropout(X)
        X = self.classifer(X)

        res = [X, hier1_logits, hier2_logits, hier3_logits]

        if self.ent_order == "eop":
            ent_order_logit = self.ent_order_logit(output_hier)
            res.append(ent_order_logit)
        if self.hier_rel_net == "rhia":
            res.append(hier1_logits_hier)
            res.append(hier2_logits_hier)
            res.append(hier3_logits_hier)
        return res

    