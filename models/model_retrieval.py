import torch
import torch.nn as nn
import torch.nn.functional as F
from models.xvlm import XVLMBase, XVLMPlusBase
import random
import numpy as np 
import timm

class CFACKCModel(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=config["mlm"], use_bbox_loss=False)
        
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
        
        #self.cnn_patch = FeatureExtractor(embed_dims=config['embed_dim'])
        #self.cnn_encoder = timm
        if config["use_cnn_feats"]:
            self.cnn_encoder = timm.create_model(config["cnn_net"], pretrained=True, num_classes=config["embed_dim"])
            if config["cnn_net"] == 'convnext_base.fb_in22k_ft_in1k':
                self.pret_model = timm.create_model(config["cnn_net"] , pretrained=True, num_classes=0)
                self.cnn_encoder = nn.Linear(self.pret_model.num_features, config["embed_dim"])
                
        #self.cnn_proj = nn.Linear(self.cnn_encoder.output_dim, config["embed_dim"])

        self.num_attention_heads = self.text_encoder.config.num_attention_heads
        self.init_params = []

        # parameter for Beta distribution of Mix Up
        self.alpha = 0.5
        # temperature params function
        #self.t_fn = Get_Scalar(0.5)  
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
    def forward(self, image1, image2, image_cnn, text1_ids, text1_atts, text2_ids, text2_atts, idx=None, mlm_inputs=None):
        
        if self.config["use_momentum"]:
            image_embeds_1, image_atts_1 = self.get_vision_embeds(image2)
            if self.config["use_cnn_feats"]:
                if self.config["cnn_net"] == 'convnext_base.fb_in22k_ft_in1k':
                    cnn_patch = self.cnn_encoder(self.pret_model(image_cnn.type(self.pret_model.stem[0].weight.dtype)))
                else:
                    cnn_patch = self.cnn_encoder(image_cnn)

        else:
            image_embeds_1, image_atts_1 = self.get_vision_embeds(image1)
            if self.config["use_cnn_feats"]:
                if self.config["cnn_net"] == 'convnext_base.fb_in22k_ft_in1k':
                    cnn_patch = self.cnn_encoder(self.pret_model(image_cnn.type(self.pret_model.stem[0].weight.dtype)))
                else:
                    cnn_patch = self.cnn_encoder(image_cnn)

        text_embeds_1 = self.get_text_embeds(text2_ids, text2_atts)
        
        image_feat_1, text_feat_1 = self.get_features(image_embeds_1, text_embeds_1)
        if self.config["use_cnn_feats"]:
            if self.config["cnn_net"] != 'convnext_base.fb_in22k_ft_in1k':
 
                cnn_patch = F.normalize(cnn_patch)

        if self.config["use_momentum"]:
            with torch.no_grad():
                self._momentum_update()

                self.temp.clamp_(0.001, 0.5)
                image_embeds_2, image_atts_2 = self.get_vision_embeds_m(image1)
                text_embeds_2 = self.get_text_embeds_m(text1_ids, text1_atts)
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
                if self.config["use_cnn_feats"]:
                    loss_cnn = self.get_contrastive_loss(image_feat_1, cnn_patch, idx=idx)
            if self.config["use_cnn_feats"]:
                loss_itc = (loss_itc_11 + loss_itc_12 + loss_itc_21 + loss_itc_22 + loss_itc_im + loss_itc_txt + loss_cnn * 0.5) / 6.5
                
            else:
                loss_itc = (loss_itc_11 + loss_itc_12 + loss_itc_21 + loss_itc_22 + loss_itc_im + loss_itc_txt) / 6

        else:
            loss_itc = self.get_contrastive_loss(image_feat_1, text_feat_1, idx=idx)
            if self.config["use_cnn_feats"]:
                loss_itc += self.get_contrastive_loss(image_feat_1, cnn_patch)
                

        # loss_itm = self.get_matching_loss_ga_aug(image_embeds_1, image_atts_1, image_feat_1, text_embeds_1, text1_atts, text_feat_1, idx=idx)
        loss_itm = self.get_matching_loss(image_embeds_1, image_atts_1, image_feat_1, text_embeds_1, text1_atts, text_feat_1, idx=idx)
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