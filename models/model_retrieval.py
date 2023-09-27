import torch
import torch.nn as nn
import torch.nn.functional as F
from models.xvlm import XVLMBase, XVLMPlusBase
import random
import numpy as np 

class SANNetwork(nn.Module):
    def __init__(self, input_size, num_heads=2, device="cuda"):
        super(SANNetwork, self).__init__()
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax2_dim0 = nn.Softmax(dim=0)
        self.device = device
        self.softmax3_dim_neg1 = nn.Softmax(dim=-1) # the last dim indicates the feature dim

        self.multi_head = nn.ModuleList([nn.Linear(input_size, input_size) for k in range(num_heads)])

    def forward_attention(self, input_space, return_softmax=False):
        placeholder = torch.zeros(input_space.shape).to(self.device)
        for k in range(len(self.multi_head)):
            if return_softmax:
                attended_matrix = self.multi_head[k](input_space)
            else:
                attended_matrix = self.softmax3_dim_neg1(self.multi_head[k](input_space)) * input_space
                #attended_matrix = self.softmax3_dim_neg1(input_space) * input_space

            placeholder = torch.add(placeholder,attended_matrix)
        placeholder /= len(self.multi_head)
        out = placeholder
        if return_softmax:
            out = self.softmax_dim1(out)
        return out

    def get_mean_attention_weights(self):
        activated_weight_matrices = []
        for head in self.multi_head:
            wm = head.weight.data
            diagonal_els = torch.diag(wm)
            activated_diagonal = self.softmax2_dim0(diagonal_els)
            activated_weight_matrices.append(activated_diagonal)
        output_mean = torch.mean(torch.stack(activated_weight_matrices, axis=0), axis=0)
        return output_mean

    def get_attention(self, x):
        return self.forward_attention(x, return_softmax=True)

    def get_softmax_hadamand_layer(self):
        return self.get_mean_attention_weights()

# HA 16 Sep 2023
class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value
    
class XVLMForRetrieval(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=config["mlm"], use_bbox_loss=False)
        
        if config['use_cross_aug']:
            self.cross_prob = config['cross_prob']
            self.cross_gene = config['cross_gene']
            # SAN network
            self.san_model = SANNetwork(self.embed_dim, 1)
        
        if config['use_momentum']:
            self.vision_encoder_m = self.build_vision_encoder(config, load_params=False)
            self.text_encoder_m, _ = self.build_text_encoder(config, vision_width=self.vision_width,
                                                                   load_text_params=False,
                                                                   use_mlm_loss=False)
            self.vision_proj_m = nn.Linear(self.vision_width, self.embed_dim)
            self.text_proj_m = nn.Linear(self.text_width, self.embed_dim)
            self.update_init_params(['vision_proj.' + n for n, _ in self.vision_proj.named_parameters()])
            self.update_init_params(['text_proj.' + n for n, _ in self.text_proj.named_parameters()])

            self.model_pairs = [(self.vision_encoder, self.vision_encoder_m),
                                (self.text_encoder, self.text_encoder_m),
                                (self.vision_proj, self.vision_proj_m),
                                (self.text_proj, self.text_proj_m)]
            self.copy_params()
            self.momentum = config['momentum']

        self.num_attention_heads = self.text_encoder.config.num_attention_heads
        self.init_params = []
        # HA 16 Sept 2023
        # parameter for Beta distribution of Mix Up
        self.alpha = 0.5
        # temperature params function
        self.t_fn = Get_Scalar(0.5)  
        # initial iteration count
        self.it = 0

        if config['use_id_loss']:
            self.classifier = nn.Linear(config['embed_dim'], config['num_classes'])
        if config['use_sdm']:
            self.temp.requires_grad = False

        self.config = config    

    def get_image_embeds_m(self, image, image_atts=None, idx_to_group_img=None, output_hidden_states=None, output_attentions=None):

        assert image.dim() == 4
        assert output_hidden_states == output_attentions

        if idx_to_group_img is None:
            image_embeds = self.vision_encoder_m(image, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            return image_embeds, image_atts  # full attention

        else:  # image < bsz
            if output_attentions or output_hidden_states:
                raise NotImplementedError

            if image_atts is None:
                image_embeds_fullatts = self.vision_encoder_m(image)

                image_embeds_fullatts = torch.gather(image_embeds_fullatts, dim=0,
                                                     index=idx_to_group_img.view(-1, 1, 1).expand(
                                                         -1, image_embeds_fullatts.shape[1],
                                                         image_embeds_fullatts.shape[2]))  # expend to bsz

                image_atts = torch.ones(image_embeds_fullatts.size()[:-1], dtype=torch.long).to(image.device)

                return image_embeds_fullatts, image_atts

            else:
                assert image_atts.size(0) == idx_to_group_img.size(0)  # bsz
                image_embeds, image_embeds_fullatts = \
                    self.vision_encoder_m(image, idx_to_group_img=idx_to_group_img, image_atts=image_atts)

                image_embeds_fullatts = torch.gather(image_embeds_fullatts, dim=0,
                                                     index=idx_to_group_img.view(-1, 1, 1).expand(
                                                         -1, image_embeds_fullatts.shape[1],
                                                         image_embeds_fullatts.shape[2]))

                return image_embeds, image_atts, image_embeds_fullatts

    def get_vision_embeds_m(self, image, image_atts=None, idx_to_group_img=None, output_hidden_states=None, output_attentions=None):
        """
        vision_embeds: cls + patch embeds
        """
        assert output_hidden_states == output_attentions

        if image.dim() == 5:  # encode video
            # image: (bsz, frame_len, c, h, w)
            assert idx_to_group_img is None, "not supported"
            return self.get_frame_embeds(image, output_hidden_states=output_hidden_states, output_attentions=output_attentions)

        assert image.dim() == 4
        return self.get_image_embeds_m(image, image_atts=image_atts, idx_to_group_img=idx_to_group_img,
                                     output_hidden_states=output_hidden_states, output_attentions=output_attentions)

    def get_text_embeds_m(self, text_ids, text_atts, output_hidden_states=None, output_attentions=None):
        assert output_hidden_states == output_attentions

        encoder = self.text_encoder_m.bert if hasattr(self.text_encoder_m, 'bert') else self.text_encoder_m

        outputs = encoder(text_ids, attention_mask=text_atts, return_dict=True, mode='text',
                          output_hidden_states=output_hidden_states, output_attentions=output_attentions)

        if output_hidden_states:
            assert len(outputs.hidden_states) == len(outputs.attentions) + 1
            return {'last_hidden_state': outputs.last_hidden_state,
                    'hidden_states': outputs.hidden_states,
                    'attentions': outputs.attentions}

        else:
            return outputs.last_hidden_state
    
    def get_features_m(self, image_embeds=None, text_embeds=None):
        if image_embeds is None:
            return F.normalize(self.text_proj_m(text_embeds[:, 0, :]), dim=-1)
        elif text_embeds is None:
            return F.normalize(self.vision_proj_m(image_embeds[:, 0, :]), dim=-1)
        else:
            return F.normalize(self.vision_proj_m(image_embeds[:, 0, :]), dim=-1), \
                   F.normalize(self.text_proj_m(text_embeds[:, 0, :]), dim=-1)
    
    #def forward(self, image1, image2, text1_ids, text1_atts, text_ids_masked, masked_pos, masked_ids, text2_ids, text2_atts, idx=None):
    def forward(self, image1, image2, text1_ids, text1_atts, text2_ids, text2_atts, idx=None, mlm_inputs=None):
        
        if self.config["use_momentum"]:
            image_embeds_1, image_atts_1 = self.get_vision_embeds(image2)
        else:
            image_embeds_1, image_atts_1 = self.get_vision_embeds(image1)

        text_embeds_1 = self.get_text_embeds(text2_ids, text2_atts)
        
        # image_embeds_1, text_embeds_1, indices = self.crossover_fm_batch_im_text(image_embeds_1, text_embeds_1)

        image_feat_1, text_feat_1 = self.get_features(image_embeds_1, text_embeds_1)
        if self.config['use_cross_aug']:
            image_feat_1 = self.cross_aug_san(image_feat_1, num_genes=self.cross_gene)
            text_feat_1 = self.cross_aug_san(text_feat_1, num_genes=self.cross_gene)

        if self.config["use_momentum"]:
            with torch.no_grad():
                self._momentum_update()

                self.temp.clamp_(0.001, 0.5)
                image_embeds_2, image_atts_2 = self.get_vision_embeds_m(image1)
                image_embeds_2 = self.crossover_fm_batch(image_embeds_2)
                text_embeds_2 = self.get_text_embeds_m(text1_ids, text1_atts)
                # image_embeds_2, text_embeds_2, indices = self.crossover_fm_batch_im_text(image_embeds_2, text_embeds_2, indices)
                # image_embeds_2.shape =  b, 577, 768
                image_feat_2, text_feat_2 = self.get_features_m(image_embeds_2, text_embeds_2)
                
                if self.config['use_cross_aug']:
                    image_feat_2 = self.cross_aug_san(image_feat_2, num_genes=self.cross_gene)
                    text_feat_2 = self.cross_aug_san(text_feat_2, num_genes=self.cross_gene)

            if self.config["use_sdm"]:
                loss_itc_11 = self.get_sdm_loss(image_feat_1, text_feat_1, pid=idx)
                loss_itc_12 = self.get_sdm_loss(image_feat_1, text_feat_2, pid=idx)
                loss_itc_21 = self.get_sdm_loss(image_feat_2, text_feat_1, pid=idx)
                loss_itc_22 = self.get_sdm_loss(image_feat_2, text_feat_2, pid=idx)
                loss_itc_im = self.get_sdm_loss(image_feat_1, image_feat_2, pid=idx)
                loss_itc_txt = self.get_sdm_loss(text_feat_1, text_feat_2, pid=idx)
            else:
                loss_itc_11 = self.get_contrastive_loss(image_feat_1, text_feat_1, idx=idx)
                loss_itc_12 = self.get_contrastive_loss(image_feat_1, text_feat_2, idx=idx)
                loss_itc_21 = self.get_contrastive_loss(image_feat_2, text_feat_1, idx=idx)
                loss_itc_22 = self.get_contrastive_loss(image_feat_2, text_feat_2, idx=idx)
                loss_itc_im = self.get_contrastive_loss(image_feat_1, image_feat_2, idx=idx)
                loss_itc_txt = self.get_contrastive_loss(text_feat_1, text_feat_2, idx=idx)


            loss_itc = (loss_itc_11 + loss_itc_12 + loss_itc_21 + loss_itc_22 + loss_itc_im + loss_itc_txt) / 6
        else:
            loss_itc = self.get_contrastive_loss(image_feat_1, text_feat_1, idx=idx)


        loss_itm = self.get_matching_loss(image_embeds_1, image_atts_1, image_feat_1, text_embeds_1, text1_atts, text_feat_1, idx=idx)
        # loss_itm = self.get_matching_loss_ga_aug(image_embeds_1, image_atts_1, image_feat_1, text_embeds_1, text1_atts, text_feat_1, idx=idx)
        #loss_itm_2 = self.get_matching_loss(image_embeds_2, image_atts_2, image_feat_2, text_embeds_1, text1_atts, text_feat_1, idx=idx)
        #loss_itm = (loss_itm_1 + loss_itm_2) / 2
        if mlm_inputs is not None:
            text_ids_masked, masked_pos, masked_ids = mlm_inputs

            loss_mlm = self.get_mlm_loss(text_ids_masked, text1_atts, image_embeds_1, image_atts_1, masked_pos, masked_ids)
            if self.config["use_id_loss"]:
                image1_logits = self.classifier(image_feat_1)
                image2_logits = self.classifier(image_feat_2)

                loss_id1 = F.cross_entropy(image1_logits, idx)
                loss_id2 = F.cross_entropy(image2_logits, idx)
                loss_id = loss_id1 + loss_id2
                
                return loss_itc, loss_itm, loss_id, loss_mlm
            
            return loss_itc, loss_itm, loss_mlm
        
        return loss_itc, loss_itm

    def cross_aug_san(self, population, num_genes=0.3): # cross 30% 
        # attn_vec = self.san_model.get_attention(torch.unsqueeze(population, 0)) # torch.unsqueeze(individual, 0))
        attn_vec = self.san_model.get_attention(population) 
        _, indices = torch.sort(attn_vec, dim=1, descending=False)
        ch_index = [random.randint(0, len(population)-1) for i in range(len(population)//2)]
        
        for index in range(len(population)//2):
            if random.random() <= self.cross_prob:
                individual = population[index]
                individual2 = population[ch_index[index]]
                        
                list_id_swap = indices[index][:int(len(indices)*num_genes)]
                for id in list_id_swap:
                    individual[id] = individual2[id] # swap gen 
                    # swap
                    population[index] = individual

        return population
    # HA 16 Sep 2023
    def mixup_one_target(self, x1, x2, alpha=1.0, is_bias=False):
        """Returns mixed inputs, mixed targets, and lambda
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        if is_bias:
            lam = max(lam, 1 - lam)

        mixed_x = lam * x1 + (1 - lam) * x2
        
        return mixed_x
    
    def get_matching_loss_ga_aug(self, image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=None):
        """
        Matching Loss with hard negatives
        
        """
        image_neg_idx, text_neg_idx = self.get_hard_negatives(image_feat, text_feat, idx=idx)

        bs = image_feat.size(0)
        image_embeds_neg = []
        image_atts_neg = []
        for b in range(bs):
            neg_idx = image_neg_idx[b]
            image_embeds_neg.append(image_embeds[neg_idx])
            image_atts_neg.append(image_atts[neg_idx])

        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        image_atts_neg = torch.stack(image_atts_neg, dim=0)

        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = text_neg_idx[b]
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts_neg, image_atts], dim=0)

        # Before Cross
        # print(f"Before Cross Embeds: {image_embeds.shape=}") # 577x768
        # print(f"Before Cross Embeds: {text_embeds.shape=}") # 40x768
        # print(f"Before Cross Embeds: {image_atts.shape=}") # 577
        # print(f"Before Cross Embeds: {text_atts.shape=}") # 40
        
        # image_embeds = self.crossover_fm_batch(image_embeds)
        # text_embeds = self.crossover_fm_batch(text_embeds)
        # print(image_embeds.shape, text_embeds.shape )
        # torch.Size([16, 577, 768]) torch.Size([16, 40, 768])
        image_embeds, text_embeds = self.crossover_fm_batch_im_text(image_embeds, text_embeds)
        # HA 19 Sep 2023
        k_cross = 5 
        last_image_embeds = []
        last_text_embeds = []
        for index in range(k_cross):
            image_embeds, text_embeds = self.crossover_fm_batch_im_text(image_embeds, text_embeds)
            if index == 0:
                last_image_embeds = image_embeds
                last_text_embeds = text_embeds
            else:
                last_image_embeds += image_embeds
                last_text_embeds += text_embeds
                
        last_image_embeds /= k_cross
        last_text_embeds /= k_cross

        # Temperature sharpening
        T = self.t_fn(self.it)
        # avg
        avg_image_prob = torch.softmax(last_image_embeds, dim=1)
        avg_image_prob = (avg_image_prob / avg_image_prob.sum(dim=-1, keepdim=True))
        avg_text_prob = torch.softmax(last_text_embeds, dim=1)
        avg_text_prob = (avg_text_prob / avg_text_prob.sum(dim=-1, keepdim=True))
        # sharpening
        cross_image_embeds = avg_image_prob ** (1 / T)
        cross_image_embeds = (cross_image_embeds / cross_image_embeds.sum(dim=-1, keepdim=True)).detach()
        cross_text_embeds = avg_text_prob ** (1 / T)
        cross_text_embeds = (cross_text_embeds / cross_text_embeds.sum(dim=-1, keepdim=True)).detach()
                        
        # Mix up
        image_embeds = self.mixup_one_target(image_embeds, cross_image_embeds, self.alpha, is_bias=True)
        text_embeds = self.mixup_one_target(text_embeds, cross_text_embeds, self.alpha, is_bias=True)

        # image_embeds = self.mutate_fm_inv_sample(image_embeds)
        # text_embeds = self.mutate_fm_inv_sample(text_embeds)
        # print(f"Before Cross Embeds: {image_embeds.shape=}")
        # print(f"Before Cross Embeds: {text_embeds.shape=}")
        # print(f"Before Cross Embeds: {image_atts.shape=}")
        # print(f"Before Cross Embeds: {text_atts.shape=}")

        cross_pos = self.get_cross_embeds(image_embeds, image_atts, text_embeds=text_embeds, text_atts=text_atts)[:, 0, :]
        cross_neg = self.get_cross_embeds(image_embeds_all, image_atts_all, text_embeds=text_embeds_all,
                                          text_atts=text_atts_all)[:, 0, :]
        # After Cross
        # print(f"After Cross Embeds: {cross_pos.shape=}")
        # print(f"After Cross Embeds: {cross_neg.shape=}")
        # exit()
        output = self.itm_head(torch.cat([cross_pos, cross_neg], dim=0))
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image_embeds.device)

        return F.cross_entropy(output, itm_labels)
    
    def get_kernel_indices(self, h, w, indices, kernel_size):
        half = kernel_size//2
        new_indices = []
        for ind in indices:
            row1 = max(ind[0]-half, 0)
            col1 = max(ind[1]-half, 0)

            row2 = min(ind[0]+half, h-1)
            col2 = min(ind[1]+half, w-1)
            # print(f"{row1=}, {row2=}, {col1=}, {col2=}")
            new_indices.extend(torch.dstack(torch.meshgrid(torch.arange(row1, row2+1), torch.arange(col1, col2+1), indexing="ij")))
        return torch.unique(torch.concat(new_indices), dim=0)

    def crossover_fm_batch(self, x, batch_indices=None):
    
        batch_size, h, w = x.size()
        p_surface = torch.rand((w,h))
        indices = torch.nonzero(p_surface < self.config["cross_prob"])
        indices = indices[:min(self.config["cross_max_features"], indices.shape[0]), :]
        
        indices_coor = self.get_kernel_indices_wh(h, w, indices, self.config["cross_kernel_size"])
        
        xx = x.clone()
        if batch_indices is None:
            xx[:,indices_coor[:, 0], indices_coor[:, 1]] = x[torch.randperm(batch_size)][:,indices_coor[:, 0], indices_coor[:, 1]] 
        else:
            xx[:,indices_coor[:, 0], indices_coor[:, 1]] = x[batch_indices][:,indices_coor[:, 0], indices_coor[:, 1]] 
        return xx
    
    def crossover_fm_batch_im_text(self, image_embeds, text_embeds, indices = None):
        
        if indices is None:
            batch_size = image_embeds.size()[0]
            indices = torch.randperm(batch_size)
        image_embeds = self.crossover_fm_batch(image_embeds, indices)
        text_embeds = self.crossover_fm_batch(text_embeds, indices)

        return image_embeds, text_embeds, indices


    def get_kernel_indices_wh(self, h, w, indices, kernel_size):
        half_row = kernel_size//2
        half_col = (kernel_size+48)//2
        new_indices = []
        for ind in indices:
            row1 = max(ind[0]-half_row, 0)
            col1 = max(ind[1]-half_col, 0)

            row2 = min(ind[0]+half_row, h-1)
            col2 = min(ind[1]+half_col, w-1)
            # print(f"{row1=}, {row2=}, {col1=}, {col2=}")
            new_indices.extend(torch.dstack(torch.meshgrid(torch.arange(row1, row2+1), torch.arange(col1, col2+1), indexing="ij")))
        return torch.unique(torch.concat(new_indices), dim=0)

    
    
    def get_kernel_indices_wh_im_text(self, h, w, indices, half_row, half_col):
        # half_row = kernel_size//2
        # half_col = (kernel_size+20)//2
        new_indices = []
        for ind in indices:
            row1 = max(ind[0]-half_row, 0)
            col1 = max(ind[1]-half_col, 0)

            row2 = min(ind[0]+half_row, h-1)
            col2 = min(ind[1]+half_col, w-1)
            # print(f"{row1=}, {row2=}, {col1=}, {col2=}")
            new_indices.extend(torch.dstack(torch.meshgrid(torch.arange(row1, row2+1), torch.arange(col1, col2+1), indexing="ij")))
        return torch.unique(torch.concat(new_indices), dim=0)
    

    def crossover_fm_batch_im_text_single(self, x, half_row, half_col, batch_indices=None):
    
        batch_size, h, w = x.size()
        p_surface = torch.rand((w,h))
        indices = torch.nonzero(p_surface < self.config["cross_prob"])
        indices = indices[:min(self.config["cross_max_features"], indices.shape[0]), :]
        
        indices_coor = self.get_kernel_indices_wh_im_text(h, w, indices, half_row, half_col)
        
        xx = x.clone()
        if indices is None:
            xx[:,indices_coor[:, 0], indices_coor[:, 1]] = x[torch.randperm(batch_size)][:,indices_coor[:, 0], indices_coor[:, 1]] 
        else:
            xx[:,indices_coor[:, 0], indices_coor[:, 1]] = x[batch_indices][:,indices_coor[:, 0], indices_coor[:, 1]] 
        return xx
    
    def crossover_fm_batch_im_text_difhw(self, image_embeds, text_embeds):
    
        batch_size = image_embeds.size()[0]
        batch_indices = torch.randperm(batch_size)
        im_cross_kernel_size = self.config["cross_kernel_size"]
        # 577x768, 40x768
        text_cross_kernel_size = int(im_cross_kernel_size / 14.25)

        image_embeds = self.crossover_fm_batch_im_text_single(image_embeds, half_row=im_cross_kernel_size//2, 
                                                              half_col=77, batch_indices=batch_indices)
        text_embeds = self.crossover_fm_batch_im_text_single(text_embeds, half_row=text_cross_kernel_size//2, half_col=77, batch_indices=batch_indices)

        return image_embeds, text_embeds
    
    def mutate_fm_inv_sample(self, x):
        batch_size, h, w = x.size()
        p = torch.rand(batch_size)
        indices = torch.arange(batch_size)[p < self.config["cross_prob"]]

        xx = x.clone()
        xx[indices] = torch.flip(x[indices], dims=[2])
        return xx

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

class XVLMPlusForRetrieval(XVLMPlusBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False, load_cross_params=False,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=False, use_bbox_loss=False)

        self.num_attention_heads = self.text_encoder.config.num_attention_heads
        self.init_params = []

    def forward(self, image, text_ids, text_atts, idx=None):
        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_text_embeds(text_ids, text_atts)

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_feat, text_feat = self.get_features(image_embeds, text_embeds)
        loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
        loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=idx)

        return loss_itc, loss_itm
