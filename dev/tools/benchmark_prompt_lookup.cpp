#include "runtime/ops/PromptLookup.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main() {
  printf("=== Benchmarking Fast CPU Prompt Lookup Decoding ===\n\n");

  splash::ops::PromptLookup pld;

  // Test 1: Synthesize a realistic code/JSON prompt with repeating patterns
  std::vector<uint32_t> promptTokens;
  // Common boilerplate tokens
  const std::vector<uint32_t> patternA = {101, 202, 303, 404, 505, 606, 707, 808};
  const std::vector<uint32_t> patternB = {999, 888, 777, 666, 555};

  for (int i = 0; i < 200; ++i) {
    promptTokens.push_back(10000 + i);
    if (i % 10 == 0) {
      promptTokens.insert(promptTokens.end(), patternA.begin(), patternA.end());
    }
    if (i % 25 == 0) {
      promptTokens.insert(promptTokens.end(), patternB.begin(), patternB.end());
    }
  }

  // 1. Benchmark Ingestion / Indexing Latency
  auto t0 = std::chrono::high_resolution_clock::now();
  constexpr int kIndexReps = 1000;
  for (int rep = 0; rep < kIndexReps; ++rep) {
    pld.indexPrompt(promptTokens, 3);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double indexUs = std::chrono::duration<double, std::micro>(t1 - t0).count() / kIndexReps;
  printf("[1] Indexing Latency: %.2f µs for %zu prompt tokens (%.2f ns/token)\n",
         indexUs, promptTokens.size(), (indexUs * 1000.0) / promptTokens.size());

  // 2. Benchmark Query / Proposal Latency
  std::array<uint32_t, 5> drafts{};
  std::array<uint32_t, 3> queryA = {101, 202, 303};

  uint32_t proposed = pld.propose(queryA, drafts, 5);
  printf("[2] Query Match Verification: proposed %u tokens\n", proposed);
  if (proposed >= 5) {
    printf("    Drafts: [%u, %u, %u, %u, %u] (Expected: [404, 505, 606, 707, 808])\n",
           drafts[0], drafts[1], drafts[2], drafts[3], drafts[4]);
    if (drafts[0] == 404 && drafts[1] == 505 && drafts[2] == 606 &&
        drafts[3] == 707 && drafts[4] == 808) {
      printf("    Result: PASS (exact pattern match)\n");
    } else {
      printf("    Result: FAIL\n");
      return 1;
    }
  } else {
    printf("    Result: FAIL (fewer than 5 proposed)\n");
    return 1;
  }

  // Measure proposal query latency across 100,000 lookups
  constexpr int kQueryReps = 100000;
  t0 = std::chrono::high_resolution_clock::now();
  uint32_t checksum = 0;
  for (int rep = 0; rep < kQueryReps; ++rep) {
    checksum += pld.propose(queryA, drafts, 5);
  }
  t1 = std::chrono::high_resolution_clock::now();
  double queryNs = std::chrono::duration<double, std::nano>(t1 - t0).count() / kQueryReps;
  printf("[3] Average Lookup Latency: %.1f ns per query (checksum=%u)\n", queryNs, checksum);

  // 3. Test Incremental appendToken
  pld.appendToken(1234);
  pld.appendToken(2345);
  pld.appendToken(3456);
  pld.appendToken(4567);
  pld.appendToken(5678);

  std::array<uint32_t, 3> queryRecent = {1234, 2345, 3456};
  proposed = pld.propose(queryRecent, drafts, 2);
  printf("[4] Incremental Token Match: proposed %u tokens\n", proposed);
  if (proposed == 2 && drafts[0] == 4567 && drafts[1] == 5678) {
    printf("    Result: PASS (exact incremental match)\n");
  } else {
    printf("    Result: FAIL\n");
    return 1;
  }

  printf("\n=== All Prompt Lookup Benchmarks PASSED Successfully ===\n");
  return 0;
}
