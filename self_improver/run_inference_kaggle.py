# ==============================================================================
# MAGIC PATCH BLOCK FOR KAGGLE P100 (FlashAttention Kapatıcı)
# ==============================================================================
import sys
import torch
import warnings
from unittest.mock import MagicMock

warnings.filterwarnings("ignore")

# 1. Kütüphaneleri kandırarak flash_attn import hatalarını sustur
sys.modules['flash_attn'] = MagicMock()
sys.modules['flash_attn.bert_padding'] = MagicMock()
sys.modules['flash_attn.bert_padding'].unpad_input = lambda tensor, mask: (tensor, None, None, None, None)

# 2. Transformers'ı SDPA kullanmaya zorla
import transformers.utils.import_utils as import_utils
import_utils.is_flash_attn_2_available = lambda: False  
from transformers import PreTrainedModel
PreTrainedModel._check_and_enable_flash_attn_2 = classmethod(lambda cls, config, *args, **kwargs: config)

# 3. Güvenli bir şekilde D2L modüllerini içe aktar
import ctx_to_lora.modeling.idefics2 as idefics2_mod
idefics2_mod.IDEFICS2_PERCEIVER_ATTENTION_CLASSES["sdpa"] = idefics2_mod.Idefics2PerceiverAttention
idefics2_mod.IDEFICS2_PERCEIVER_ATTENTION_CLASSES["eager"] = idefics2_mod.Idefics2PerceiverAttention

# 4. Resampler'ın boyut hatasını atla, temiz SDPA akışı kur
def _safe_resampler_forward(self, context, attention_mask=None, position_ids=None):
    bsz = context.shape[0]
    latents = self.latents.expand(bsz, -1, -1)
    projected_inputs = self.proj_in(context)
    latents = self.encoder(
        context=projected_inputs,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cu_seq_lens_q=None,
        cu_seq_lens_k=None,
        max_length_q=None,
        max_length_k=None,
        latents=latents,
    )
    return self.proj_out(latents)

idefics2_mod.Idefics2PerceiverResampler.forward = _safe_resampler_forward
idefics2_mod.Idefics2PerceiverResampler._use_flash_attention_2 = False

# 5. Standart attention'ı FlashAttention parametrelerinden arındır
_orig_attn_forward = idefics2_mod.Idefics2PerceiverAttention.forward
def _safe_attn_forward(self, latents, is_cross_attn=True, context=None, **kwargs):
    clean_kwargs = {k: v for k, v in kwargs.items() if k not in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]}
    if not is_cross_attn or context is None:
        return _orig_attn_forward(self, latents, latents, **clean_kwargs)
    return _orig_attn_forward(self, latents, context, **clean_kwargs)

idefics2_mod.Idefics2PerceiverAttention.forward = _safe_attn_forward
# ==============================================================================


# ==============================================================================
# SİZİN PAYLAŞTIĞINIZ D2L KODUNUN BİREBİR AYNISI
# ==============================================================================
# caveat: this interface only supports non-batched inputs
# for batched inference please see `src/ctx_to_lora/modeling/hypernet.py`
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

# model loading
checkpoint_path = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
state_dict = torch.load(checkpoint_path, weights_only=False)
model = ModulatedPretrainedModel.from_state_dict(
    state_dict, train=False, use_sequence_packing=False
)
model.reset()
tokenizer = get_tokenizer(model.base_model.name_or_path)

# prepare data
doc = open("data/sakana_wiki.txt", "r").read()
chat = [{"role": "user", "content": "Tell me about Sakana AI."}]
chat_ids = tokenizer.apply_chat_template(
    chat,
    add_special_tokens=False,
    return_attention_mask=False,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)


# calls after internalization will be influenced by internalized info
model.internalize(doc)

outputs = model.generate(input_ids=chat_ids, max_new_tokens=512)
print("--- MODEL CIKTISI ---")
print(tokenizer.decode(outputs[0]))

# remove internalized info
# model.reset()

# without internalized info, the model will halucinate
# outputs = model.generate(input_ids=chat_ids, max_new_tokens=512)
# print(tokenizer.decode(outputs[0]))
