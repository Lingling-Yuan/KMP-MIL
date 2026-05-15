import torch
import torch.nn as nn
import torch.nn.functional as F

from conch import get_tokenizer, tokenize
from .pfc import PrototypeConditionedFeatureCalibration
from .tc import TextualCalibration


def merge_parameter(cfg, param):
    if cfg["opt_name"] != "adam":
        raise NotImplementedError("Invalid optimizer")
    return torch.cat([p for p in param], dim=0)


class CONCHPromptEncoder(nn.Module):
    """Reuse the CONCH text transformer for learned prompt embeddings."""

    def __init__(self, coca_model):
        super().__init__()
        coca_text_model = coca_model.text

        self.pad_id = coca_text_model.pad_id
        assert self.pad_id == 0, "CONCH prompt encoding assumes pad_id = 0."
        self.heads = coca_text_model.heads
        self.positional_embedding = coca_text_model.positional_embedding
        self.attn_mask = coca_text_model.attn_mask
        self.transformer = coca_text_model.transformer
        self.ln_final = coca_text_model.ln_final
        self.cls_emb = coca_text_model.cls_emb
        self.text_projection = coca_text_model.text_projection
        self.token_embedding = coca_text_model.token_embedding
        self.text_config = {
            "max_num_tokens": 127,
            "embedding_dim": self.token_embedding.embedding_dim,
            "embedding_dtype": self.token_embedding.weight.dtype,
        }

    def build_cls_mask(self, text, cast_dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        return torch.repeat_interleave(additive_mask, self.heads, 0)

    def _repeat(self, tensor, n):
        return tensor.reshape(1, 1, -1).repeat(n, 1, 1)

    def forward(self, prompts_embedding, prompts_pseudo_tokens):
        cast_dtype = self.transformer.get_cast_dtype()
        device = prompts_embedding.device
        seq_len = prompts_embedding.shape[1]
        x = prompts_embedding.to(cast_dtype)

        prompts_pseudo_tokens = prompts_pseudo_tokens.to(device)
        attn_mask = self.attn_mask.to(device)
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(prompts_pseudo_tokens, cast_dtype)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)

        if self.cls_emb is not None:
            pooled = self.ln_final(x[:, -1])
        else:
            x = self.ln_final(x)
            pooled = x[torch.arange(x.shape[0]), prompts_pseudo_tokens.argmax(dim=-1)]

        if self.text_projection is not None:
            pooled = pooled @ self.text_projection
        return pooled


class PromptLearner(nn.Module):
    """Build KMP prompt embeddings and class text prototypes from textual_knowledge."""

    def __init__(self, cfg, base_model, text_encoder, device, prompt, current_class_prompts):
        super().__init__()
        self.cfg = cfg
        if cfg["base_model_arch"] != "CONCH":
            raise NotImplementedError("Please specify a valid architecture.")

        self.embedding_dim = base_model.text.ln_final.weight.shape[0]
        self.prompt = prompt
        text_encoder.eval()

        self.tokenized_prompts, embedding = self._get_embedding(
            base_model, device, num_placeholder=cfg["prompt_length"]
        )
        self.embedding_prefix = embedding[:, :1, :]
        self.embedding_suffix = embedding[:, 1 + cfg["prompt_length"]:, :]

        count = current_class_prompts["count"]
        tokenized_texts, text_embedding = self._get_embedding(
            base_model,
            device,
            classes=current_class_prompts["class_prompts"],
        )
        with torch.no_grad():
            class_feature_matrix = text_encoder(text_embedding, tokenized_texts)
        self.class_prompt_feature = self._average_class_prompts(class_feature_matrix, count)

    def _average_class_prompts(self, class_feature_matrix, count):
        ends = []
        total = 0
        for item in count:
            total += int(item)
            ends.append(total)

        start = 0
        features = []
        for end in ends:
            features.append(torch.mean(class_feature_matrix[start:end], dim=0, keepdim=True))
            start = end
        return torch.cat(features, dim=0)

    def _get_embedding(self, base_model, device, num_placeholder=0, classes=None):
        prompt_prefix = " ".join(["x"] * num_placeholder)
        if classes is None:
            prompts = [prompt_prefix + "."]
        elif prompt_prefix:
            prompts = [f"{prompt_prefix} {name}." for name in classes]
        else:
            prompts = [f"{name}." for name in classes]

        if self.cfg["base_model_arch"] != "CONCH":
            raise NotImplementedError("Please specify a valid architecture.")

        tokenized_prompts = tokenize(get_tokenizer(), prompts)[:, :-1]
        with torch.no_grad():
            embedding = base_model.text.token_embedding(tokenized_prompts.to(device)).type(base_model.dtype)
        return tokenized_prompts, embedding

    def forward(self, indices, mini_batch):
        merged_prompt = merge_parameter(self.cfg, self.prompt)
        embedding_core = merged_prompt[indices]

        embedding_prefix = self.embedding_prefix.unsqueeze(0).repeat(
            mini_batch, self.cfg["match_size"], 1, 1
        )
        embedding_suffix = self.embedding_suffix.unsqueeze(0).repeat(
            mini_batch, self.cfg["match_size"], 1, 1
        )
        embedding = torch.cat([embedding_prefix, embedding_core, embedding_suffix], dim=2)
        embedding = embedding.view(mini_batch * self.cfg["match_size"], -1, self.embedding_dim)

        tokenized_prompts = self.tokenized_prompts.unsqueeze(0).repeat(
            mini_batch, self.cfg["match_size"], 1
        )
        tokenized_prompts = tokenized_prompts.view(mini_batch * self.cfg["match_size"], -1)
        return embedding, tokenized_prompts


class KMPMIL(nn.Module):
    """KMP-MIL with KMP retrieval, PFC visual calibration, and TC text calibration."""

    def __init__(
        self,
        cfg,
        base_model,
        device,
        key,
        prompt,
        tc_residuals,
        current_class_prompts,
        pfc_gamma_delta_pool=None,
        pfc_beta_pool=None,
        past_keys=None,
        current_descriptors=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.dtype = base_model.dtype
        self.logit_scale = base_model.logit_scale
        self.device = device

        self.key = key
        if current_descriptors:
            print(f"[KMP] Initializing keys with {len(current_descriptors)} semantic descriptors...")
            with torch.no_grad():
                tokens = tokenize(get_tokenizer(), current_descriptors).to(device)
                descriptor_embeddings = base_model.encode_text(tokens)
                descriptor_embeddings = F.normalize(descriptor_embeddings, dim=-1)
                self.key = nn.ParameterList([
                    nn.Parameter(descriptor_embeddings[i:i + 1].type(self.dtype))
                    for i in range(len(current_descriptors))
                ])

        self.prompt = prompt
        self.tc = TextualCalibration(cfg, tc_residuals)
        self.textual_calibration = self.tc

        if cfg["base_model_arch"] != "CONCH":
            raise NotImplementedError("Please specify a valid architecture.")
        self.text_encoder = CONCHPromptEncoder(base_model)
        self.prompt_learner = PromptLearner(
            cfg, base_model, self.text_encoder, device, prompt, current_class_prompts
        )

        self.past_keys = []
        if past_keys is not None:
            self.past_keys = [k.to(device).detach() for k in past_keys]

        self.use_pfc = (
            bool(cfg.get("use_pfc", True))
            and pfc_gamma_delta_pool is not None
            and pfc_beta_pool is not None
        )
        if self.use_pfc:
            self.pfc = PrototypeConditionedFeatureCalibration(
                cfg=cfg,
                device=device,
                dtype=self.dtype,
                gamma_delta_pool=pfc_gamma_delta_pool,
                beta_pool=pfc_beta_pool,
            )
        else:
            self.pfc = None

    def _query_prototype_pool(self, x_list, mini_batch, eval):
        q_vec_list = []
        for i in range(mini_batch):
            x_list[i] = x_list[i].type(self.dtype)
            if self.cfg["pooling"] == "max":
                q_vec, _ = torch.max(x_list[i], dim=1)
            elif self.cfg["pooling"] == "mean":
                q_vec = torch.mean(x_list[i], dim=1)
            else:
                raise NotImplementedError("invalid pooling method")
            q_vec_list.append(q_vec)

        q_vecs = torch.cat(q_vec_list, dim=0)
        q_vecs = q_vecs / (q_vecs.norm(dim=-1, keepdim=True) + 1e-12)

        current_key = F.normalize(merge_parameter(self.cfg, self.key), dim=-1)
        cos_sim = q_vecs @ current_key.t()
        indices = cos_sim.topk(k=self.cfg["match_size"], dim=1, largest=True).indices

        proto_scores = cos_sim.gather(dim=1, index=indices)

        if eval:
            matching_loss = None
            routing_loss = None
        else:
            matched_key = current_key[indices]
            q_vecs_rep = q_vecs.unsqueeze(1).expand(-1, self.cfg["match_size"], -1)
            matching_loss = 1 - ((q_vecs_rep * matched_key) / (mini_batch * self.cfg["match_size"])).sum()

            if self.past_keys:
                past_keys = F.normalize(torch.cat(self.past_keys, dim=0).to(self.device), dim=-1)
                max_past = (q_vecs @ past_keys.t()).max(dim=1).values
                max_cur = cos_sim.max(dim=1).values
                margin = float(self.cfg.get("route_margin", 0.05))
                temp = max(float(self.cfg.get("route_temperature", 0.07)), 1e-6)
                routing_loss = F.softplus((max_past - max_cur + margin) / temp).mean()
            else:
                routing_loss = torch.tensor(0.0, device=self.device)

        return indices, matching_loss, proto_scores, routing_loss

    def _get_bag_feature(self, x_list, indices, mini_batch):
        embedding, tokenized_prompts = self.prompt_learner(indices, mini_batch)
        prototype_features = self.text_encoder(embedding, tokenized_prompts)
        prototype_features = prototype_features / (prototype_features.norm(dim=-1, keepdim=True) + 1e-12)
        prototype_features = prototype_features.view(mini_batch, self.cfg["match_size"], -1)

        bag_feature_list = []
        for i, x in enumerate(x_list):
            x = x.squeeze()
            x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-12)
            sim_matrix = self.cfg["csm_logit_scale"] * x_norm @ prototype_features[i].t()
            weights = torch.softmax(sim_matrix, dim=0)
            bag_feature_list.append(torch.mean(weights.t() @ x, dim=0, keepdim=True))

        bag_feature = torch.cat(bag_feature_list, dim=0)
        bag_feature = bag_feature / (bag_feature.norm(dim=-1, keepdim=True) + 1e-12)
        return bag_feature.unsqueeze(1)

    def get_patch_attention(self, x_list, indices):
        mini_batch = len(x_list)
        embedding, tokenized_prompts = self.prompt_learner(indices, mini_batch)
        prototype_features = self.text_encoder(embedding, tokenized_prompts)
        prototype_features = prototype_features / (prototype_features.norm(dim=-1, keepdim=True) + 1e-12)
        prototype_features = prototype_features.view(mini_batch, self.cfg["match_size"], -1)

        attn_list = []
        for i, x in enumerate(x_list):
            x = x.squeeze()
            x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-12)
            sim_matrix = self.cfg["csm_logit_scale"] * x_norm @ prototype_features[i].t()
            attn_list.append(torch.softmax(sim_matrix, dim=0).detach())
        return attn_list

    def _compute_class_sim_loss(self, enhanced_class_feature):
        features = enhanced_class_feature[0]
        n_cls = features.shape[0]
        cos_sim = features @ features.permute(1, 0)
        cos_sim = cos_sim + 1
        return cos_sim[~torch.eye(n_cls, dtype=torch.bool, device=self.device)].mean()

    def forward(self, x_list, eval=False):
        mini_batch = len(x_list)
        indices, matching_loss, proto_scores, routing_loss = self._query_prototype_pool(
            x_list, mini_batch, eval
        )

        if self.use_pfc:
            x_list, pfc_reg_loss, pfc_debug = self.pfc(x_list, indices, proto_scores)
        else:
            pfc_reg_loss, pfc_debug = None, None

        bag_feature = self._get_bag_feature(x_list, indices, mini_batch)
        class_feature = self.tc(self.prompt_learner.class_prompt_feature)
        class_feature = class_feature.unsqueeze(0).repeat(mini_batch, 1, 1)

        bag_feature_norm = F.normalize(bag_feature, p=2, dim=-1)
        class_feature_norm = F.normalize(class_feature, p=2, dim=-1)
        logits = self.logit_scale.exp() * (bag_feature_norm * class_feature_norm).sum(-1)

        if eval:
            return logits.float(), indices

        loss_dict = {
            "matching_loss": matching_loss.float(),
            "routing_loss": routing_loss.float(),
            "class_sim_loss": self._compute_class_sim_loss(class_feature_norm).float(),
            "pfc_reg_loss": pfc_reg_loss.float() if self.use_pfc else torch.tensor(0.0, device=self.device),
        }
        return logits.float(), loss_dict, indices
