#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#ifdef USE_Q8
#include "model_q8_kv.h"
#elif defined(USE_KVCACHE_V2)
#include "model_kvcache_v2.h"
#elif defined(USE_KVCACHE)
#include "model_kvcache.h"
#elif defined(USE_SIMD)
#include "model_simd.h"
#else
#include "model.h"
#endif

int main(int argc, char **argv) {
  const char *model_path = "data/model.bin";
  const char *vocab_path = "data/vocab.bin";
  const char *ids_str = nullptr;
  int max_length = 50;
  float temperature = 0.0f;
  bool machine_mode = false;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--ids") == 0 && i + 1 < argc)
      ids_str = argv[++i];
    else if (strcmp(argv[i], "--max_length") == 0 && i + 1 < argc)
      max_length = atoi(argv[++i]);
    else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc)
      temperature = atof(argv[++i]);
    else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
      model_path = argv[++i];
    else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc)
      vocab_path = argv[++i];
    else if (strcmp(argv[i], "--machine") == 0)
      machine_mode = true;
    else if (strcmp(argv[i], "--help") == 0) {
      printf("Usage: ./gpt2 --ids \"id1 id2 ...\" [--max_length N] "
             "[--temperature T] [--machine]\n");
      return 0;
    }
  }

  if (!ids_str) {
    fprintf(stderr, "Error: --ids required\n");
    return 1;
  }

  std::vector<int> input_ids;
  std::istringstream iss(ids_str);
  int id;
  while (iss >> id)
    input_ids.push_back(id);

  // Suppress prints in machine mode by redirecting load messages to stderr
  if (!machine_mode) {
    printf("Input IDs (%zu tokens):", input_ids.size());
    for (int i : input_ids)
      printf(" %d", i);
    printf("\n");
  }

  GPT2 model = load_gpt2(model_path);
  Vocab vocab = load_vocab(vocab_path);

  if (!machine_mode) {
    printf("Input: %s\n", decode(vocab, input_ids).c_str());
#ifdef USE_SIMD
    printf("\n--- Generating [SIMD] (max_length=%d, temperature=%.2f) ---\n",
           max_length, temperature);
#else
    printf("\n--- Generating [NAIVE] (max_length=%d, temperature=%.2f) ---\n",
           max_length, temperature);
#endif
  }

  std::vector<int> all_ids = input_ids;
  auto t_start = std::chrono::high_resolution_clock::now();

  for (int step = 0; step < max_length; step++) {
    float *logits = gpt2_forward(model, all_ids.data(), (int)all_ids.size());
    int next_id =
        (temperature <= 0.0f)
            ? sample_argmax(logits, model.config.vocab_size)
            : sample_temp(logits, model.config.vocab_size, temperature);
    all_ids.push_back(next_id);
    if (!machine_mode) {
      printf("%s", vocab.tokens[next_id].c_str());
      fflush(stdout);
    }
    if (next_id == 50256)
      break;
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();
  int gen_tokens = (int)all_ids.size() - (int)input_ids.size();

  if (machine_mode) {
    // Machine-readable: token IDs + timing (avoids newline issues in text)
    printf("MACHINE_IDS:");
    for (size_t i = 0; i < all_ids.size(); i++)
      printf("%s%d", i ? " " : "", all_ids[i]);
    printf("\nMACHINE_TIME:%.6f\n", elapsed);
  } else {
    printf("\n--- Done ---\n");
    printf("Full output: %s\n", decode(vocab, all_ids).c_str());
    printf("Generated %d tokens in %.3f sec (%.1f tok/sec)\n", gen_tokens,
           elapsed, gen_tokens / elapsed);
  }

  free_gpt2(model);
  return 0;
}
