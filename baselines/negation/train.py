"""
networks for DANN, CDAN in adapting negation
"""
from tqdm import tqdm
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

import tent


def dann_adapt(args, src_encoder, encoder, discriminator, classifier,
    src_loader, tgt_loader):
    """
    first add KD, IM loss,
    use a half of train set to build GMM, then resample as src features?
    split train set into 2 parts as src and tgt
    """
    src_encoder.eval()
    encoder.train()
    classifier.eval()
    discriminator.train()

    loss_domain = nn.CrossEntropyLoss()
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')
    optimizer_e = optim.Adam(encoder.parameters(), lr=args.learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.learning_rate)
    len_dataloader = min(len(src_loader), len(tgt_loader))

    for epoch in range(args.num_epochs):
        # zip source and target data pair
        pbar = tqdm(zip(src_loader, tgt_loader))
        for i, (src_batch, tgt_batch) in enumerate(pbar):
            src_batch = tuple(t.to(args.device) for t in src_batch)
            src_inputs = {'input_ids': src_batch[0],
                          'attention_mask': src_batch[1],
                          'token_type_ids': None}
            tgt_batch = tuple(t.to(args.device) for t in tgt_batch)
            tgt_inputs = {'input_ids': tgt_batch[0],
                          'attention_mask': tgt_batch[1],
                          'token_type_ids': None}
            # if len(src_feat) != len(tgt_batch[0]):
            #     continue
            src_feat = src_encoder(**src_inputs)[1]

            p = float(
                i + epoch * len_dataloader) / args.num_epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero gradients for optimizers
            optimizer_e.zero_grad()
            # optimizer_c.zero_grad()
            optimizer_d.zero_grad()

            # extract and concat features
            # s_class_output = classifier(src_feat)
            s_reverse_feat = ReverseLayerF.apply(src_feat, alpha)
            s_domain_output = discriminator(s_reverse_feat)
            # loss_s_label = loss_class(s_class_output, src_label.float())
            s_domain_label = torch.zeros(src_feat.size()[0]).long().to(args.device)
            loss_s_domain = loss_domain(s_domain_output, s_domain_label)

            tgt_feat = encoder(**tgt_inputs)[1]
            t_reverse_feat = ReverseLayerF.apply(tgt_feat, alpha)
            t_domain_output = discriminator(t_reverse_feat)
            t_domain_label = torch.ones(tgt_feat.size()[0]).long().to(args.device)
            loss_t_domain = loss_domain(t_domain_output, t_domain_label)
            loss_d = loss_s_domain + loss_t_domain
            tgt_outputs = classifier(encoder(**tgt_inputs)[0])
            if args.kd:  # KD loss
                t = args.temperature
                # src_tgt_feat = src_encoder(**tgt_inputs)[1]
                with torch.no_grad():
                    src_prob = F.softmax(classifier(src_encoder(**tgt_inputs)[0]) / t, dim=-1)
                tgt_prob = F.log_softmax(tgt_outputs / t, dim=-1)
                kd_loss = kl_div_loss(tgt_prob, src_prob.detach()) * t * t
                # loss += kd_loss

            if args.ent:  # IM loss
                softmax_out = nn.Softmax(dim=1)(tgt_outputs)
                entropy_loss = torch.mean(entropy(softmax_out))  # loss_ent
                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                    entropy_loss -= torch.sum(-msoftmax * torch.log(
                        msoftmax + 1e-6))  # loss_ent + loss_div
                im_loss = entropy_loss * args.ent_par
                # loss += im_loss
            loss = 0* loss_d + 0*kd_loss + 1*im_loss

            loss.backward()
            optimizer_e.step()
            # optimizer_c.step()
            optimizer_d.step()

            if i % args.log_step == 0:
                desc = f"Epoch [{epoch}/{args.num_epochs}] Step [{i}/{len_dataloader}] " \
                       f"l_s_dom={loss_s_domain.item():.3f} " \
                       f"l_t_dom={loss_t_domain.item():.3f}"
                pbar.set_description(desc=desc)

        # evaluate(args, encoder, classifier, tgt_all_loader)

    return encoder


def cdan_adapt(args, src_encoder, encoder, discriminator, classifier,
    src_loader, tgt_loader):
    """
    CDAN adapt w self-distillation
    """
    encoder.train()
    classifier.eval()
    discriminator.train()

    loss_domain = nn.BCELoss()
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')
    optimizer_e = optim.SGD(encoder.parameters(), lr=args.learning_rate,
                            weight_decay=5e-3, momentum=0.9)
    optimizer_d = optim.SGD(discriminator.parameters(), lr=args.learning_rate,
                            weight_decay=5e-3, momentum=0.9)
    len_dataloader = min(len(src_loader), len(tgt_loader))
    
    for epoch in range(args.num_epochs):
        pbar = tqdm(zip(src_loader, tgt_loader))
        for i, (src_batch, tgt_batch) in enumerate(pbar):
            src_batch = tuple(t.to(args.device) for t in src_batch)
            src_inputs = {'input_ids': src_batch[0],
                          'attention_mask': src_batch[1],
                          'token_type_ids': None}
            # src_label = src_batch[3]
            tgt_batch = tuple(t.to(args.device) for t in tgt_batch)
            tgt_inputs = {'input_ids': tgt_batch[0],
                          'attention_mask': tgt_batch[1],
                          'token_type_ids': None}
            # if len(src_batch[0]) != len(tgt_batch[0]):
            #     continue

            # zero gradients for optimizers
            optimizer_e.zero_grad()
            optimizer_d.zero_grad()

            # extract and concat features
            src_feat0, src_feat1 = encoder(**src_inputs)
            tgt_feat0, tgt_feat1 = encoder(**tgt_inputs)
            feat0 = torch.cat((src_feat0, tgt_feat0), 0)
            feat1 = torch.cat((src_feat1, tgt_feat1), 0)
            # s_class_output = classifier(src_feat)
            # loss_s_label = loss_class(s_class_output,
            #                           src_label.float())  # maybe change s_class_output to before softmax

            class_output = classifier(feat0)
            op_out = torch.bmm(class_output.unsqueeze(2), feat1.unsqueeze(1))
            ad_out = discriminator(
                op_out.view(-1, class_output.size(1) * feat1.size(1)))
            dc_target = torch.from_numpy(np.array([[1]] * src_feat1.size()[0]
                                                  + [[0]] * tgt_feat1.size()[
                                                      0])).float().to(args.device)
            loss_d = loss_domain(ad_out, dc_target)
            tgt_outputs = classifier(tgt_feat0)

            if args.kd:  # KD loss
                t = args.temperature
                # src_tgt_feat = src_encoder(**tgt_inputs)[1]
                with torch.no_grad():
                    src_prob = F.softmax(classifier(src_encoder(**tgt_inputs)[0]) / t, dim=-1)
                tgt_prob = F.log_softmax(tgt_outputs / t, dim=-1)
                kd_loss = kl_div_loss(tgt_prob, src_prob.detach()) * t * t
                # loss += kd_loss

            if args.ent:  # IM loss
                softmax_out = nn.Softmax(dim=1)(tgt_outputs)
                entropy_loss = torch.mean(entropy(softmax_out))  # loss_ent
                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                    entropy_loss -= torch.sum(-msoftmax * torch.log(
                        msoftmax + 1e-6))  # loss_ent + loss_div
                im_loss = entropy_loss * args.ent_par
                # loss += im_loss
            loss = 0* loss_d + 0.5*kd_loss + 0.5*im_loss

            loss.backward()

            optimizer_e.step()
            optimizer_d.step()

            if i % args.log_step == 0:
                desc = f"Epoch [{epoch}/{args.num_epochs}] Step [{i}/{len_dataloader}] " \
                       f"l_d={loss_d.item():.4f} "
                pbar.set_description(desc=desc)

    return encoder


def tent_adapt(args, combined_model):
    """
    tent adapt
    https://github.com/DequanWang/tent/blob/master/cifar10c.py
    """
    combined_model = tent.configure_model(combined_model)
    params, param_names = tent.collect_params(combined_model)
    optimizer = setup_optimizer(args, params)
    combined_model = tent.Tent(combined_model, optimizer,
                            steps=1, episodic=False)
    # print(f"model for adaptation: %s", combined_model)
    print(f"params for adaptation: %s", param_names)
    print(f"optimizer for adaptation: %s", optimizer)
    return combined_model


def setup_optimizer(args, params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if args.optim_method == 'Adam':
        return optim.Adam(params,
                    lr=args.learning_rate,
                    betas=(0.9, 0.999),
                    weight_decay=0)
    elif args.optim_method == 'SGD':
        return optim.SGD(params,
                   lr=args.learning_rate,
                   momentum=0.9,
                   dampening=0,
                   weight_decay=0,
                   nesterov=True)
    else:
        raise NotImplementedError


class AdversarialNetworkDann(nn.Module):
    """
    Domain discriminator for DANN
    """

    def __init__(self, args):
        super(AdversarialNetworkDann, self).__init__()
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module(
            "d_fc1", nn.Linear(args.input_dim, args.hidden_dim)
        )
        self.domain_classifier.add_module("d_bn1", nn.BatchNorm1d(args.hidden_dim))
        self.domain_classifier.add_module("d_relu1", nn.ReLU(True))
        self.domain_classifier.add_module(
            "d_fc2", nn.Linear(args.hidden_dim, args.num_labels)
        )
        self.domain_classifier.add_module("d_softmax", nn.LogSoftmax(dim=1))

    def forward(self, reverse_feature):
        domain_output = self.domain_classifier(reverse_feature)
        return domain_output


class AdversarialNetworkCdan(nn.Module):
    """
    Domain discriminator for CDAN
    """

    def __init__(self, args):
        super(AdversarialNetworkCdan, self).__init__()
        self.ad_layer1 = nn.Linear(args.input_dim * args.num_labels, args.hidden_dim)
        self.ad_layer2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.ad_layer3 = nn.Linear(args.hidden_dim, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(self.init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = self.calc_coeff(
            self.iter_num, self.high, self.low, self.alpha, self.max_iter
        )
        x = x * 1.0
        x.register_hook(self.grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def calc_coeff(self, iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
        return np.float(
            2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter))
            - (high - low)
            + low
        )

    def init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)
        elif classname.find("Linear") != -1:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def grl_hook(self, coeff):
        def fun1(grad):
            return -coeff * grad.clone()

        return fun1


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def entropy(inputs):
    # bs = inputs.size(0)
    ent = -inputs * torch.log(inputs + 1e-5)
    ent = torch.sum(ent, dim=1)
    return ent


class BatchNorm(nn.Module):
    """
    add a batchnorm layer on features for tent
    """
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.layer = nn.BatchNorm1d(input_dim, affine=True)
    
    def forward(self, x):
        return self.layer(x)