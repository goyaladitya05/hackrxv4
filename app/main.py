from fastapi import FastAPI
from app.routes import router

'''
import warnings

# Suppress specific deprecation warning from transformers/torch internals
warnings.filterwarnings(
    "ignore",
    message="`encoder_attention_mask` is deprecated and will be removed in version 4.55.0 for `BertSdpaSelfAttention.forward`.",
    category=FutureWarning,
    module="torch.nn.modules.module"
)
'''
app = FastAPI(title="HackRx LLM Query-Retrieval System")

app.include_router(router, prefix="/api/v1")
