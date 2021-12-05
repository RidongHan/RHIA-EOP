import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time
import numpy as np
from Net import MyModel, LabelSmoothLoss
from sklearn import metrics
from data_loader import my_data_loader
from config import config
from utils import AverageMeter

def get_hier_rel(hier_rel, scope):
    new_rel = []
    index = 0
    for s in scope:
        new_rel += [hier_rel[index]] * (s[1]-s[0])
        index += 1
    return torch.stack(new_rel)

def train(train_loader, test_loader, opt):
    if opt['model'] == 'RHIA-EOP':
        model = MyModel(
            train_loader.dataset.vec_save_dir, 
            train_loader.dataset.hier1_rel_num(),
            train_loader.dataset.hier2_rel_num(),
            train_loader.dataset.hier3_rel_num(), 
            lambda_pcnn=opt['lambda_pcnn'], 
            ent_order=opt['ent_order'],
            hier_rel_net=opt['hier_rel_net']
            )
            
        hier1_criterion_S = LabelSmoothLoss(smoothing=0.1)
        hier2_criterion_S = LabelSmoothLoss(smoothing=0.1)
        hier3_criterion_S = LabelSmoothLoss(smoothing=0.1)
        if opt['ent_order'] == "eop":
            print("consider entity order loss !")
            criterion_ent_order = LabelSmoothLoss(smoothing=0.01)
        if opt['hier_rel_net'] == "rhia":
            hier1_criterion_hier = LabelSmoothLoss(smoothing=0.1)
            hier2_criterion_hier = LabelSmoothLoss(smoothing=0.1)
            hier3_criterion_hier = LabelSmoothLoss(smoothing=0.1)
        criterion = nn.CrossEntropyLoss() 
    
    if torch.cuda.is_available():
        model = model.cuda()

    print(model)
    
    optimizer = optim.SGD(model.parameters(), lr=opt['lr'], weight_decay=1e-5)  
    # momentum=0.9
    not_best_count = 0
    best_auc = 0
    if not os.path.exists(opt['save_dir']):
        os.mkdir(opt['save_dir'])
    ckpt = os.path.join(opt['save_dir'], 'model.pth.tar')
    for epoch in range(opt['epoch']):
        model.train()
        print("\n=== Epoch %d train ===" % epoch)
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()
        avg_pos_acc = AverageMeter()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                for d in range(len(data)):
                    data[d] = data[d].cuda()
            if opt['model'] == 'RHIA-EOP':
                word, pos1, pos2, ent_order, ent_pos, index1, index2, ent1, ent2, mask, length, scope, hier1_rel, hier2_rel, rel = data
                res = model(word, pos1, pos2, ent_order, ent_pos, index1, index2, ent1, ent2, mask, scope, length)
                output, output_hier1_S, output_hier2_S, output_hier3_S = res[0: 4]
                if opt['ent_order'] == "eop":
                    ent_order_logit = res[4]
                    if opt['hier_rel_net'] == "rhia":
                        output_hier1_hier, output_hier2_hier, output_hier3_hier = res[5: ]
                else:
                    if opt['hier_rel_net'] == "rhia":
                        output_hier1_hier, output_hier2_hier, output_hier3_hier = res[4: ]

                    
                hier1_rel = get_hier_rel(hier1_rel, scope)  # 一个bag 一个label
                hier2_rel = get_hier_rel(hier2_rel, scope)
                hier3_rel = get_hier_rel(rel, scope)
                loss_hier1_S = hier1_criterion_S(output_hier1_S, hier1_rel)
                loss_hier2_S = hier2_criterion_S(output_hier2_S, hier2_rel)
                loss_hier3_S = hier3_criterion_S(output_hier3_S, hier3_rel)
                if opt['ent_order'] == "eop":
                    loss_ent_order = criterion_ent_order(ent_order_logit, ent_order)

            loss = criterion(output, rel)
            if opt['model'] == 'RHI-EOP':
                loss += loss_hier1_S
                loss += loss_hier2_S
                loss += loss_hier3_S
                if opt['ent_order'] == "eop":
                    loss += opt['eop_w'] * loss_ent_order

            _, pred = torch.max(output, -1)
            acc = (pred == rel).sum().item() / rel.shape[0]
            pos_total = (rel != 0).sum().item()
            pos_correct = ((pred == rel) & (rel != 0)).sum().item()
            if pos_total > 0:
                pos_acc = pos_correct / pos_total
            else:
                pos_acc = 0
            # Log
            avg_loss.update(loss.item(), 1)
            avg_acc.update(acc, 1)
            avg_pos_acc.update(pos_acc, 1)
            sys.stdout.write('\rstep: %d | loss: %f, acc: %f, pos_acc: %f'%(i, avg_loss.avg, avg_acc.avg, avg_pos_acc.avg))
            sys.stdout.flush()
            # Optimize
            loss.backward()
            optimizer.step()

        if (epoch + 1) % opt['val_iter'] == 0 and avg_pos_acc.avg > 0.3:
            print("\n=== Epoch %d val ===" % epoch)
            y_true, y_pred = valid(test_loader, model, opt)
            auc = metrics.average_precision_score(y_true, y_pred)
            print("\n[TEST] auc: {}".format(auc))
            if auc > best_auc:
                print("Best result!")
                best_auc = auc
                torch.save({'state_dict': model.state_dict()}, ckpt)
                not_best_count = 0
            else:
                not_best_count += 1
            if not_best_count >= opt['early_stop']:
                break


def valid(test_loader, model, opt):
    model.eval()
    avg_acc = AverageMeter()
    avg_pos_acc = AverageMeter()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if torch.cuda.is_available():
                data = [x.cuda() for x in data]
            if opt['model'] == 'RHIA-EOP':
                word, pos1, pos2, ent_order, ent_pos, index1, index2, ent1, ent2, mask, length, scope, hier1_rel, hier2_rel, rel = data
                res = model(word, pos1, pos2, ent_order, ent_pos, index1, index2, ent1, ent2, mask, scope, length)
                output, output_hier1_S, output_hier2_S, output_hier3_S = res[0: 4]
                if opt['ent_order'] == "eop":
                    ent_order_logit = res[4]
                    if opt['hier_rel_net'] == "rhia":
                        output_hier1_hier, output_hier2_hier, output_hier3_hier = res[5: ]
                else:
                    if opt['hier_rel_net'] == "rhia":
                        output_hier1_hier, output_hier2_hier, output_hier3_hier = res[4: ]

            output = torch.softmax(output, -1)
            label = rel.argmax(-1)
            _, pred = torch.max(output, -1)
            acc = (pred == label).sum().item() / label.shape[0]
            pos_total = (label != 0).sum().item()
            pos_correct = ((pred == label) & (label != 0)).sum().item()
            if pos_total > 0:
                pos_acc = pos_correct / pos_total
            else:
                pos_acc = 0
            # Log
            avg_acc.update(acc, 1)
            avg_pos_acc.update(pos_acc, 1)
            sys.stdout.write('\rstep: %d | acc: %f, pos_acc: %f'%(i, avg_acc.avg, avg_pos_acc.avg))
            sys.stdout.flush()
            y_true.append(rel[:, 1:])  # exclude NA
            y_pred.append(output[:, 1:])
    y_true = torch.cat(y_true).reshape(-1).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).reshape(-1).detach().cpu().numpy()
    return y_true, y_pred

def test(test_loader, opt):
    print("=== Test ===")
    # Load model
    save_dir = opt['save_dir']
    if opt['model'] == 'RHIA-EOP':
        model = MyModel(
            train_loader.dataset.vec_save_dir, 
            train_loader.dataset.hier1_rel_num(),
            train_loader.dataset.hier2_rel_num(),
            train_loader.dataset.hier3_rel_num(), 
            lambda_pcnn=opt['lambda_pcnn'], 
            ent_order=opt['ent_order'],
            hier_rel_net=opt['hier_rel_net']
            )
    if torch.cuda.is_available():
        model = model.cuda()
    state_dict = torch.load(os.path.join(save_dir, 'model.pth.tar'))['state_dict']
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)

    y_true, y_pred = valid(test_loader, model, opt)
    # AUC
    auc = metrics.average_precision_score(y_true, y_pred)
    print("\n[TEST] auc: {}".format(auc))
    # P@N
    order = np.argsort(-y_pred)
    p100 = (y_true[order[:100]]).mean() * 100
    p200 = (y_true[order[:200]]).mean() * 100
    p300 = (y_true[order[:300]]).mean() * 100
    print("P@100: {0:.1f}, P@200: {1:.1f}, P@300: {2:.1f}, Mean: {3:.1f}".
          format(p100, p200, p300, (p100 + p200 + p300) / 3))
    # PR
    order = np.argsort(y_pred)[::-1]
    correct = 0.
    total = y_true.sum()
    precision = []
    recall = []
    for i, o in enumerate(order):
        correct += y_true[o]
        precision.append(float(correct) / (i + 1))
        recall.append(float(correct) / total)
    precision = np.array(precision)
    recall = np.array(recall)
    print("Saving result")
    np.save(os.path.join(save_dir, 'precision.npy'), precision)
    np.save(os.path.join(save_dir, 'recall.npy'), recall)
    return y_true, y_pred


if __name__ == '__main__':
    print("reading parameters: config.py")
    opt = vars(config())
    # # torch.backends.cudnn.enabled = True
    # # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    if opt['model'] == 'RHIA-EOP':
        train_loader = my_data_loader(opt['train'], opt, shuffle=True, training=True)
        test_loader = my_data_loader(opt['test'], opt, shuffle=False, training=False)
    train(train_loader, test_loader, opt)
    y_true, y_pred = test(test_loader, opt)

