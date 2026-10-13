#include "official_dsa_decode_v57.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T>
static std::vector<T> read_binary(
    const std::string &path,
    std::size_t expected_count) {

    std::ifstream file(path.c_str(), std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open " + path);
    }

    std::vector<T> values(expected_count);
    file.read(
        reinterpret_cast<char *>(values.data()),
        expected_count * sizeof(T));

    if (!file
        || static_cast<std::size_t>(file.gcount())
            != expected_count * sizeof(T)) {
        throw std::runtime_error(
            "Incorrect file size or incomplete read: " + path);
    }

    return values;
}

static std::vector<odsa_i8_t> convert_i8(
    const std::vector<std::int8_t> &input) {

    std::vector<odsa_i8_t> output(input.size());

    for (std::size_t index = 0;
         index < input.size();
         ++index) {
        output[index] = static_cast<int>(input[index]);
    }

    return output;
}


static std::vector<odsa_wide_t> pack_wide(
    const std::vector<std::int8_t> &input) {

    if (input.size() % 64 != 0) {
        throw std::runtime_error(
            "Wide input size is not divisible by 64");
    }

    std::vector<odsa_wide_t> output(input.size() / 64);

    for (std::size_t word = 0;
         word < output.size();
         ++word) {

        odsa_wide_t packed = 0;

        for (int lane = 0; lane < 64; ++lane) {
            std::uint8_t raw =
                static_cast<std::uint8_t>(
                    input[word * 64 + lane]);

            packed.range(
                lane * 8 + 7,
                lane * 8) = raw;
        }

        output[word] = packed;
    }

    return output;
}

int main() {
    static const int CONTEXT_LENGTH = 4096;

    const char *fixture_environment =
        std::getenv("OFFICIAL_FIXTURE_DIR");

    if (fixture_environment == 0) {
        std::cerr
            << "ERROR: OFFICIAL_FIXTURE_DIR is not set\n";
        return 2;
    }

    const std::string directory(fixture_environment);

    try {
        std::vector<odsa_wide_t> index_q = pack_wide(
            read_binary<std::int8_t>(
                directory + "/index_q.bin",
                ODSA_INDEX_HEADS * ODSA_INDEX_DIM));

        std::vector<odsa_wide_t> index_k = pack_wide(
            read_binary<std::int8_t>(
                directory + "/index_k.bin",
                CONTEXT_LENGTH * ODSA_INDEX_DIM));

        std::vector<odsa_i8_t> index_weights = convert_i8(
            read_binary<std::int8_t>(
                directory + "/index_weights.bin",
                ODSA_INDEX_HEADS));

        std::vector<odsa_wide_t> q_latent = pack_wide(
            read_binary<std::int8_t>(
                directory + "/q_latent.bin",
                ODSA_ATTN_HEADS * ODSA_LATENT_DIM));

        std::vector<odsa_wide_t> q_rope = pack_wide(
            read_binary<std::int8_t>(
                directory + "/q_rope.bin",
                ODSA_ATTN_HEADS * ODSA_ROPE_DIM));

        std::vector<odsa_wide_t> latent_cache = pack_wide(
            read_binary<std::int8_t>(
                directory + "/latent_cache.bin",
                CONTEXT_LENGTH * ODSA_LATENT_DIM));

        std::vector<odsa_wide_t> pe_cache = pack_wide(
            read_binary<std::int8_t>(
                directory + "/pe_cache.bin",
                CONTEXT_LENGTH * ODSA_ROPE_DIM));

        std::vector<odsa_wide_t> value_projection = pack_wide(
            read_binary<std::int8_t>(
                directory + "/value_projection.bin",
                ODSA_ATTN_HEADS
                    * ODSA_VALUE_DIM
                    * ODSA_LATENT_DIM));

        std::vector<std::int32_t> reference_selected =
            read_binary<std::int32_t>(
                directory + "/selected.bin",
                ODSA_TOPK);

        std::vector<float> reference_output =
            read_binary<float>(
                directory + "/output_scaled_quant_f32.bin",
                ODSA_OUTPUT_SIZE);

        std::vector<odsa_wide_t> selected_packed(ODSA_TOPK / 16);
        std::vector<odsa_wide_t> output_packed(ODSA_OUTPUT_SIZE / 16);
        std::vector<odsa_token_t> selected(ODSA_TOPK);
        std::vector<float> output(ODSA_OUTPUT_SIZE, 0.0f);

        std::cout << "Running complete official-config kernel\n";
        std::cout << "Context length: " << CONTEXT_LENGTH << "\n";
        std::cout << "Index heads: " << ODSA_INDEX_HEADS << "\n";
        std::cout << "Top-k: " << ODSA_TOPK << "\n";
        std::cout << "Attention heads: " << ODSA_ATTN_HEADS << "\n";

        official_dsa_decode(
            index_q.data(),
            index_k.data(),
            index_k.data(),
            index_k.data(),
            index_k.data(),
            index_weights.data(),
            q_latent.data(),
            q_rope.data(),
            latent_cache.data(),
            latent_cache.data(),
            pe_cache.data(),
            value_projection.data(),
            CONTEXT_LENGTH,
            selected_packed.data(),
            output_packed.data());

        for (int entry = 0; entry < ODSA_TOPK; ++entry) {
            selected[entry] = (odsa_token_t)(unsigned int)
                selected_packed[entry / 16].range(
                    (entry % 16) * 32 + 31, (entry % 16) * 32);
        }

        for (int index = 0; index < ODSA_OUTPUT_SIZE; ++index) {
            const int head = index / ODSA_VALUE_DIM;
            const int value_dimension = index % ODSA_VALUE_DIM;
            union { float f; unsigned int u; } conv;
            conv.u = (unsigned int)output_packed[value_dimension].range(
                head * 32 + 31, head * 32);
            output[index] = conv.f;
        }

        std::vector<int> actual_selected(ODSA_TOPK);
        std::vector<int> expected_selected(ODSA_TOPK);

        for (int entry = 0; entry < ODSA_TOPK; ++entry) {
            actual_selected[entry] =
                static_cast<unsigned int>(selected[entry]);
            expected_selected[entry] =
                reference_selected[entry];
        }

        std::sort(
            actual_selected.begin(),
            actual_selected.end());

        std::sort(
            expected_selected.begin(),
            expected_selected.end());

        int selection_errors = 0;

        for (int entry = 0; entry < ODSA_TOPK; ++entry) {
            if (actual_selected[entry]
                != expected_selected[entry]) {
                if (selection_errors < 10) {
                    std::cerr
                        << "Selected mismatch at " << entry
                        << ": expected "
                        << expected_selected[entry]
                        << ", got "
                        << actual_selected[entry]
                        << "\n";
                }
                ++selection_errors;
            }
        }

        double maximum_absolute_error = 0.0;
        double squared_error = 0.0;
        double dot_product = 0.0;
        double actual_norm = 0.0;
        double reference_norm = 0.0;

        for (int index = 0;
             index < ODSA_OUTPUT_SIZE;
             ++index) {

            double actual = output[index];
            double expected = reference_output[index];
            double difference = actual - expected;
            double absolute_error = std::fabs(difference);

            if (absolute_error > maximum_absolute_error) {
                maximum_absolute_error = absolute_error;
            }

            squared_error += difference * difference;
            dot_product += actual * expected;
            actual_norm += actual * actual;
            reference_norm += expected * expected;
        }

        double rmse = std::sqrt(
            squared_error / ODSA_OUTPUT_SIZE);

        double cosine_similarity =
            dot_product
            / std::sqrt(actual_norm * reference_norm);

        std::cout
            << "Top-k mismatches: "
            << selection_errors << "\n";

        std::cout
            << "Maximum absolute output error: "
            << maximum_absolute_error << "\n";

        std::cout
            << "Output RMSE: "
            << rmse << "\n";

        std::cout
            << "Output cosine similarity: "
            << cosine_similarity << "\n";

        bool selection_pass = selection_errors == 0;

        bool output_pass =
            maximum_absolute_error <= 2.0
            && cosine_similarity >= 0.999999;

        if (!selection_pass || !output_pass) {
            std::cerr << "OFFICIAL CONFIGURATION CSIM FAIL\n";
            return 1;
        }

        std::cout << "OFFICIAL CONFIGURATION CSIM PASS\n";
        return 0;

    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 3;
    }
}
