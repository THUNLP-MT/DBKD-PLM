import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertForSequenceClassification, BertForMultipleChoice, GPT2LMHeadModel

class DBKD_Classification_Model(BertForSequenceClassification):
    def __init__(self, config,
                 kd_alpha=1.0,
                 ce_alpha=1.0,
                 sigma=10,
                 temperature=5.0,
                 kl_kd=False,
                 writer=None,
                 bsize=32,
                 log_interval=500,
                 strategy="none",
                 soft_label_path=""):
        super().__init__(config)
        self.bsize = bsize
        self.num_labels = config.num_labels
        self.kd_alpha = kd_alpha
        self.ce_alpha = ce_alpha
        self.sigma = sigma
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.kl_kd = kl_kd
        self.temperature = temperature
        self.writer = writer
        self.step = 0
        self.log_interval = log_interval
        self.log_record = {
            "ce_loss": 0,
            "kd_loss": 0,
            "entropy": 0
        }
        self.strategy = strategy
        self.label_smoothing = 0.1
        if self.strategy in ["smoothing", "none", "random"]:
            self.soft_label = None
        else:
            self.soft_label = torch.load(soft_label_path).cuda()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                logger=None,
                **kwargs):
        if 'idx' in kwargs:
            idx = kwargs['idx']
        kd_loss = None
        if self.training:
            self.step += 1
            student_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)

            if self.strategy == "none":
                kd_loss = None
            elif self.strategy == "standard" or self.strategy == "surrogate":
                with torch.no_grad():
                    teacher_logits = self.soft_label[idx]
                if self.kl_kd:
                    kd_loss = self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                           F.softmax(teacher_logits / self.temperature, dim=1)) * self.temperature ** 2
                else:
                    kd_loss = self.mse_loss(student_logits, teacher_logits)
            elif self.strategy == "DBKD":
                with torch.no_grad():
                    teacher_logits = self.soft_label[idx]
                    teacher_logits = teacher_logits - teacher_logits.mean(dim=-1).unsqueeze(-1)
                if self.kl_kd:
                    kd_loss = self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                           F.softmax(teacher_logits * self.sigma / self.temperature, dim=1)) * self.temperature ** 2
                else:
                    assert False
            elif self.strategy == "noise-smooth":
                smooth_factor = 0.01
                with torch.no_grad():
                    teacher_probs = self.soft_label[idx]
                teacher_probs = teacher_probs / teacher_probs.sum(dim=-1).unsqueeze(-1)
                teacher_probs = (1 - smooth_factor) * teacher_probs + smooth_factor
                teacher_probs = teacher_probs ** (1 / self.temperature)
                teacher_probs = teacher_probs / teacher_probs.sum(dim=-1).unsqueeze(-1)
                if self.kl_kd:
                    kd_loss = self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                           teacher_probs) * self.temperature ** 2
                else:
                    assert False
            elif self.strategy == "hard-smooth":
                with torch.no_grad():
                    teacher_probs = self.soft_label[idx]
                    teacher_probs = F.one_hot(teacher_probs.argmax(dim=-1), self.num_labels).float()
                    smooth_probs = (1.0 - self.label_smoothing) * teacher_probs + self.label_smoothing / self.num_labels
                    smooth_probs = smooth_probs ** (1 / self.temperature)
                    smooth_probs = smooth_probs / smooth_probs.sum(dim=-1).unsqueeze(-1)
                if self.kl_kd:
                    kd_loss = self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                           smooth_probs.float()) * self.temperature ** 2
                else:
                    assert False
            elif self.strategy == "random":
                # random_logits = torch.empty(student_logits.shape).normal_(mean=torch.mean(student_logits).item(),
                #                                                           std=torch.std(student_logits).item())
                random_logits = torch.empty(student_logits.shape).normal_()
                if self.kl_kd:
                    kd_loss = self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                           F.softmax(random_logits.cuda() / self.temperature, dim=1)) * self.temperature ** 2
                else:
                    kd_loss = self.mse_loss(student_logits, random_logits)
            elif self.strategy == "hard":
                with torch.no_grad():
                    teacher_probs = self.soft_label[idx]
                    labels = teacher_probs.argmax(dim=-1)
                    kd_loss = None
                    # teacher_probs = F.one_hot(teacher_probs.argmax(dim=-1), self.num_labels)
                # if self.kl_kd:
                #     kd_loss = self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                #                            teacher_probs.float()) * self.temperature ** 2
                # else:
                #     assert False
            else:
                assert False

        else:  # use student model for inference
            student_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                ce_loss = self.ce_alpha * loss_fct(student_logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                ce_loss = self.ce_alpha * loss_fct(student_logits.view(-1, self.num_labels), labels.view(-1))
            loss = ce_loss

            if kd_loss is not None:
                if self.step % self.log_interval == 0 and self.log_record["ce_loss"] > 0:
                    self.writer.add_scalar('train/CE_loss', self.log_record["ce_loss"]/self.log_interval, self.step)
                    self.writer.add_scalar('train/kd_loss', self.log_record["kd_loss"]/self.log_interval, self.step)
                    self.log_record["ce_loss"] = 0
                    self.log_record["kd_loss"] = 0
                loss += self.kd_alpha * kd_loss

        output = (student_logits,) + student_outputs[2:]
        return ((loss,) + output) if loss is not None else output

class DBKD_MultiChoice_Model(BertForMultipleChoice):
    def __init__(self, config,
                 kd_alpha=1.0,
                 ce_alpha=1.0,
                 temperature=5.0,
                 sigma=10,
                 kl_kd=False,
                 writer=None,
                 bsize=32,
                 log_interval=500,
                 strategy="none",
                 soft_label_path=""):
        super().__init__(config)
        self.bsize = bsize
        self.num_labels = config.num_labels
        self.kd_alpha = kd_alpha
        self.ce_alpha = ce_alpha
        self.sigma = sigma
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.kl_kd = kl_kd
        self.temperature = temperature
        self.writer = writer
        self.step = 0
        self.log_interval = log_interval
        self.log_record = {
            "ce_loss": 0,
            "kd_loss": 0,
            "entropy": 0
        }
        self.strategy = strategy
        self.label_smoothing = 0.1
        if self.strategy in ["smoothing", "none", "random"]:
            self.soft_label = None
        else:
            self.soft_label = torch.load(soft_label_path).cuda()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                logger=None,
                **kwargs):
        if 'idx' in kwargs:
            idx = kwargs['idx']
        kd_loss = None
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        if self.training:
            self.step += 1
            student_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)
            student_logits = student_logits.view(-1, num_choices)

            if self.strategy == "none":
                kd_loss = None
            elif self.strategy == "standard" or self.strategy == "surrogate":
                with torch.no_grad():
                    teacher_logits = self.soft_label[idx]
                if self.kl_kd:
                    kd_loss = self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                           F.softmax(teacher_logits / self.temperature, dim=1)) * self.temperature ** 2
                else:
                    kd_loss = self.mse_loss(student_logits, teacher_logits)
            elif self.strategy == "DBKD":
                with torch.no_grad():
                    teacher_logits = self.soft_label[idx]
                    teacher_logits = teacher_logits - teacher_logits.mean(dim=-1).unsqueeze(-1)
                if self.kl_kd:
                    kd_loss = self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                           F.softmax(teacher_logits * self.sigma / self.temperature, dim=1)) * self.temperature ** 2
                else:
                    assert False
            elif self.strategy == "hard-smooth":
                with torch.no_grad():
                    teacher_probs = self.soft_label[idx]
                    teacher_probs = F.one_hot(teacher_probs.argmax(dim=-1), self.num_labels).float()
                    smooth_probs = (1.0 - self.label_smoothing) * teacher_probs + self.label_smoothing / self.num_labels
                    smooth_probs = smooth_probs ** (1 / self.temperature)
                    smooth_probs = smooth_probs / smooth_probs.sum(dim=-1).unsqueeze(-1)
                if self.kl_kd:
                    kd_loss = self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                           smooth_probs.float()) * self.temperature ** 2
                else:
                    assert False
            elif self.strategy == "random":
                random_logits = torch.empty(student_logits.shape).normal_(mean=torch.mean(student_logits).item(),
                                                                          std=torch.std(student_logits).item())
                if self.kl_kd:
                    kd_loss = self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                           F.softmax(random_logits.cuda() / self.temperature, dim=1)) * self.temperature ** 2
                else:
                    kd_loss = self.mse_loss(student_logits, random_logits)
            elif self.strategy == "hard" or self.strategy == "hard_aug":
                with torch.no_grad():
                    teacher_probs = self.soft_label[idx]
                    labels = teacher_probs.argmax(dim=-1)
                    kd_loss = None
            else:
                assert False

        else:  # use student model for inference
            student_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)
            student_logits = student_logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                ce_loss = self.ce_alpha * loss_fct(student_logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                ce_loss = self.ce_alpha * loss_fct(student_logits.view(-1, self.num_labels), labels.view(-1))
            loss = ce_loss

            if kd_loss is not None:
                if self.step % self.log_interval == 0 and self.log_record["ce_loss"] > 0:
                    self.writer.add_scalar('train/CE_loss', self.log_record["ce_loss"]/self.log_interval, self.step)
                    self.writer.add_scalar('train/kd_loss', self.log_record["kd_loss"]/self.log_interval, self.step)
                    self.log_record["ce_loss"] = 0
                    self.log_record["kd_loss"] = 0
                loss += self.kd_alpha * kd_loss

        output = (student_logits,) + student_outputs[2:]
        return ((loss,) + output) if loss is not None else output


class DBKD_MultiChoice_GPT_Model(GPT2LMHeadModel):
    def __init__(self, config,
                 tokenizer,
                 kd_alpha=1.0,
                 ce_alpha=1.0,
                 temperature=5.0,
                 sigma=10,
                 kl_kd=False,
                 writer=None,
                 bsize=32,
                 log_interval=500,
                 strategy="none",
                 soft_label_path=""):
        super().__init__(config)
        self.bsize = bsize
        self.num_labels = config.num_labels
        self.kd_alpha = kd_alpha
        self.ce_alpha = ce_alpha
        self.sigma = sigma
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.kl_loss_agent = nn.KLDivLoss(reduction='none')
        self.kl_kd = kl_kd
        self.temperature = temperature
        self.writer = writer
        self.step = 0
        self.log_interval = log_interval
        self.log_record = {
            "ce_loss": 0,
            "kd_loss": 0,
            "entropy": 0
        }
        self.strategy = strategy
        self.label_smoothing = 0.1
        self.tokenizer = tokenizer
        if self.strategy in ["smoothing", "none", "random"]:
            self.soft_label = None
        else:
            self.soft_label = torch.load(soft_label_path).cuda()
        self.choice_idx = [self.tokenizer.encode(choice)[0] for choice in [' A', ' B', ' C', ' D']]

    def soft_label2logits(self, soft_label, size, neg_value=-1e5):
        logits = torch.full((soft_label.shape[0], size), neg_value)
        for idx in range(len(logits)):
            for choice, label in zip(self.choice_idx, soft_label[idx]):
                logits[idx][choice] = label
        return logits.to(soft_label.device)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs):
        if 'idx' in kwargs:
            idx = kwargs['idx']

        kd_loss = None
        if self.training:
            self.step += 1
            input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
            attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
            position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
            student_outputs = self.transformer(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            pooled_output = student_outputs[0]
            student_logits = self.lm_head(pooled_output)
            length_vec = attention_mask.sum(dim=-1)
            student_logits = torch.stack([student_l[length-1] for student_l, length in zip(student_logits, length_vec)])
            labels = torch.stack([lbl[length - 1] for lbl, length in zip(labels, length_vec)])

            if self.strategy == "none":
                kd_loss = None
            elif self.strategy == "standard":
                with torch.no_grad():
                    teacher_logits = self.soft_label[idx]
                if self.kl_kd:
                    kd_loss = self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                           F.softmax(teacher_logits / self.temperature, dim=1)) * self.temperature ** 2
                else:
                    kd_loss = self.mse_loss(student_logits, teacher_logits)
            elif self.strategy == "DBKD":
                with torch.no_grad():
                    teacher_logits = self.soft_label2logits(self.soft_label[idx],
                                                            student_logits.shape[-1])
                    teacher_logits = teacher_logits - teacher_logits.mean(dim=-1).unsqueeze(-1)
                if self.kl_kd:
                    kd_loss = self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                           F.softmax(teacher_logits * self.sigma / self.temperature, dim=1)) * self.temperature ** 2
                else:
                    assert False
            elif self.strategy == "hard-smooth":
                with torch.no_grad():
                    teacher_probs = self.soft_label2logits(self.soft_label[idx],
                                                            student_logits.shape[-1])
                    teacher_probs = F.one_hot(teacher_probs.argmax(dim=-1), student_logits.shape[-1]).float()
                    smooth_probs = (1.0 - self.label_smoothing) * teacher_probs + self.label_smoothing / student_logits.shape[-1]
                    smooth_probs = smooth_probs ** (1 / self.temperature)
                    smooth_probs = smooth_probs / smooth_probs.sum(dim=-1).unsqueeze(-1)
                if self.kl_kd:
                    kd_loss = self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                           smooth_probs.float()) * self.temperature ** 2
                else:
                    assert False
            elif self.strategy == "random":
                random_logits = torch.empty(student_logits.shape).normal_(mean=torch.mean(student_logits).item(),
                                                                          std=torch.std(student_logits).item())
                if self.kl_kd:
                    kd_loss = self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                           F.softmax(random_logits.cuda() / self.temperature, dim=1)) * self.temperature ** 2
                else:
                    kd_loss = self.mse_loss(student_logits, random_logits)
            elif self.strategy == "hard" or self.strategy == "hard_aug":
                with torch.no_grad():
                    teacher_probs = self.soft_label2logits(self.soft_label[idx],
                                                            student_logits.shape[-1])
                    labels = teacher_probs.argmax(dim=-1)
                    kd_loss = None
            elif self.strategy == "surrogate":
                with torch.no_grad():
                    teacher_probs = self.soft_label2logits(self.soft_label[idx],
                                                            student_logits.shape[-1])
                    teacher_probs = teacher_probs ** (1 / self.temperature)
                if self.kl_kd:
                    kd_loss = self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                           teacher_probs) * self.temperature ** 2
                else:
                    kd_loss = self.mse_loss(student_logits, teacher_probs)
            else:
                assert False

        else:  # use student model for inference
            input_ids = input_ids.view(1, input_ids.size(-1)) if input_ids is not None else None
            attention_mask = attention_mask.view(1, attention_mask.size(-1)) if attention_mask is not None else None
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
            position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
            student_outputs = self.transformer(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            pooled_output = student_outputs[0]
            student_logits = self.lm_head(pooled_output)
            length_vec = attention_mask.sum(dim=-1)
            student_logits = torch.stack([student_l[length - 1] for student_l, length in zip(student_logits, length_vec)])
            student_logits = student_logits[:, self.choice_idx]

        loss = None
        if labels is not None and self.training:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                ce_loss = self.ce_alpha * loss_fct(student_logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                ce_loss = self.ce_alpha * loss_fct(student_logits.view(labels.shape[0], -1), labels.view(-1))
            loss = ce_loss

            if kd_loss is not None:
                if self.step % self.log_interval == 0 and self.log_record["ce_loss"] > 0:
                    self.writer.add_scalar('train/CE_loss', self.log_record["ce_loss"]/self.log_interval, self.step)
                    self.writer.add_scalar('train/kd_loss', self.log_record["kd_loss"]/self.log_interval, self.step)
                    self.log_record["ce_loss"] = 0
                    self.log_record["kd_loss"] = 0
                loss += self.kd_alpha * kd_loss

        output = (student_logits,) + student_outputs[2:]
        # import pdb
        # pdb.set_trace()
        return ((loss,) + output) if loss is not None else output
