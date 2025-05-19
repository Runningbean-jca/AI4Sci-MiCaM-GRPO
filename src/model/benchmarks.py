from typing import List
from types import SimpleNamespace

from guacamol.distribution_learning_benchmark import (
    KLDivBenchmark,
    NoveltyBenchmark,
    UniquenessBenchmark,
    ValidityBenchmark,
)
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.utils.chemistry import is_valid

from model.mydataclass import BenchmarkResults

# ✅ 替换导入 fcd_benchmark
from fcd_torch import FCD


class QuickBenchGenerator(DistributionMatchingGenerator):
    def __init__(self, generator: DistributionMatchingGenerator, number_samples: int = 10000, max_tries: int = 20):
        self.generator = generator

        max_samples = max_tries * number_samples
        number_already_sampled = 0
        number_unique_molecules = 0

        unique_molecules: List[str] = []
        all_molecules: List[str] = []

        iter = 0
        while number_unique_molecules < number_samples and number_already_sampled < max_samples:
            iter += 1
            remaining_to_sample = number_samples - number_unique_molecules
            samples = generator.generate(number_samples=remaining_to_sample)
            number_already_sampled += remaining_to_sample
            for m in samples:
                if is_valid(m):
                    if m not in unique_molecules:
                        unique_molecules.append(m)
                        number_unique_molecules += 1
            all_molecules += samples

        assert len(unique_molecules) >= number_samples

        self.molecules = all_molecules
        self.pt = 0

    def generate(self, number_samples: int) -> List[str]:
        samples: List[str] = []
        while len(samples) < number_samples:
            samples.append(self.molecules[self.pt])
            self.pt = (self.pt + 1) % len(self.molecules)
        return samples


class QuickBenchmark(object):
    def __init__(self, training_set: List[str], num_samples: int = 10000) -> None:
        self.num_samples = num_samples
        self.valid_bench = ValidityBenchmark(number_samples=num_samples)
        self.uniq_bench = UniquenessBenchmark(number_samples=num_samples)
        self.novel_bench = NoveltyBenchmark(number_samples=num_samples, training_set=training_set)
        self.kl_bench = KLDivBenchmark(number_samples=num_samples, training_set=training_set)


        # ✅ 替换 FrechetBenchmark 为 FCD
        import torch
        self.fcd_metric = FCD(device="cuda" if torch.cuda.is_available() else "cpu")
        self.training_set = training_set

    def assess_model(self, generator):
        quickbenchgenerator = QuickBenchGenerator(generator, number_samples=self.num_samples)

        valid_result = self.valid_bench.assess_model(quickbenchgenerator)
        uniq_result = self.uniq_bench.assess_model(quickbenchgenerator)
        novel_result = self.novel_bench.assess_model(quickbenchgenerator)
        kl_result = self.kl_bench.assess_model(quickbenchgenerator)

        # ✅ 使用 fcd_torch 的接口计算 FCD
        generated_smiles = quickbenchgenerator.molecules
        fcd_score = self.fcd_metric(generated_smiles, self.training_set)
        
        fcd_result = SimpleNamespace(score=fcd_score)

        return BenchmarkResults(
            validity=valid_result,
            uniqueness=uniq_result,
            novelty=novel_result,
            kl_div=kl_result,
            fcd=fcd_result,
        )
