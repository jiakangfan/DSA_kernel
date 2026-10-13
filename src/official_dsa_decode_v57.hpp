#ifndef OFFICIAL_DSA_DECODE_V5_HPP
#define OFFICIAL_DSA_DECODE_V5_HPP

#include <ap_int.h>
#include "official_dsa_config.hpp"

typedef ap_int<8>    odsa_i8_t;
typedef ap_uint<17>  odsa_token_t;
typedef ap_uint<512> odsa_wide_t;

void official_dsa_decode(
    const odsa_wide_t *index_q,
    const odsa_wide_t *index_k0,
    const odsa_wide_t *index_k1,
    const odsa_wide_t *index_k2,
    const odsa_wide_t *index_k3,
    const odsa_i8_t *index_weights,
    const odsa_wide_t *q_latent,
    const odsa_wide_t *q_rope,
    const odsa_wide_t *latent_cache0,
    const odsa_wide_t *latent_cache1,
    const odsa_wide_t *pe_cache,
    const odsa_wide_t *value_projection,
    int context_length,
    odsa_wide_t *selected_out,
    odsa_wide_t *output);

#endif
