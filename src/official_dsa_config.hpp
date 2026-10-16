#ifndef OFFICIAL_DSA_CONFIG_HPP
#define OFFICIAL_DSA_CONFIG_HPP

static const int ODSA_MAX_CONTEXT    = 131072;

static const int ODSA_INDEX_HEADS    = 64;
static const int ODSA_INDEX_DIM      = 128;
static const int ODSA_TOPK           = 2048;

static const int ODSA_ATTN_HEADS     = 16;
static const int ODSA_LATENT_DIM     = 512;
static const int ODSA_ROPE_DIM       = 64;
static const int ODSA_EFFECTIVE_QK   = 576;
static const int ODSA_VALUE_DIM      = 128;
static const int ODSA_OUTPUT_SIZE    =
    ODSA_ATTN_HEADS * ODSA_VALUE_DIM;

static const int ODSA_HEAP_LEVELS    = 11;

#endif
