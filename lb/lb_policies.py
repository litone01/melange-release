import random
from datetime import datetime

NUM_INPUT_LENS = 6
NUM_OUTPUT_LENS = 6

def get_size_bucket_idx(size):
    if size <= 25:
        return 0
    elif size <= 100:
        return 1
    elif size <= 250:
        return 2
    elif size <= 500:
        return 3
    elif size <= 1000:
        return 4
    else:
        return 5

GPUS_LLAMA7b = {
   "A10G" : {
      "tputs": [
                  [33.96,12.35,3.98,1.46,0.44,0.13],
                  [25.49,7.85,2.99,1.12,0.39,0.11],
                  [11.47,5.5,2.15,0.86,0.32,0.11],
                  [6.16,2.62,1.32,0.6,0.26,0.11],
                  [3.1,1.51,0.77,0.38,0.17,0.07],
                  [1.61,0.81,0.42,0.22,0.11,0.04]
               ],
   },
   "A100" : {
      "tputs": [
                  [46.7,21.65,10.38,4.68,1.7,0.51],
                  [44.48,21.41,9.77,3.97,1.57,0.45],
                  [33.7,17.13,7.58,3.29,1.28,0.52],
                  [18.59,9.19,4.74,2.42,1.04,0.47],
                  [9.85,5.56,3.01,1.62,0.73,0.3],
                  [5.22,3.08,1.72,0.97,0.48,0.2],
               ],
   }
}

class LBPolicy:
    def __init__(self):
        self.backends = []

    def SetBackends(self, backends):
        self.backends = backends


class RoundRobin(LBPolicy):
    def __init__(self):
        super().__init__()
        self.currentIndex = -1

    def LoadBalance(self):
        self.currentIndex = (self.currentIndex + 1) % len(self.backends)
        return self.backends[self.currentIndex]
     

class WeightedRandom_InputAndOutput(LBPolicy):
    def __init__(self):
        super().__init__()
        self.ComputeWeights()

    def ComputeWeights(self):
        for backend in self.backends:
            backend["weights"] = []
        for i in range(NUM_INPUT_LENS):
            for backend in self.backends:
                backend["weights"].append([])
            for j in range(NUM_OUTPUT_LENS):
                aggregate_tput = sum([GPUS_LLAMA7b[backend["gpu_type"]]["tputs"][i][j] for backend in self.backends])
                for backend in self.backends:
                    backend["weights"][i].append(GPUS_LLAMA7b[backend["gpu_type"]]["tputs"][i][j] / aggregate_tput)

        # For debugging:
        # for backend in self.backends:
        #     print(f'Weights: {backend["weights"]}')
        #     print(backend["weights"][0][0])
        # for i in range(NUM_INPUT_LENS):
        #     for j in range(NUM_OUTPUT_LENS):
        #         aggregate_weight = sum([backend["weights"][i][j] for backend in self.backends])
        #         print(f'Weight sum (should be 1): aggregate_weight')

    def LoadBalance(self, prompt_length, output_length):
        # Needed for InputAndOutput policy
        assert(output_length)

        prompt_idx = get_size_bucket_idx(prompt_length)
        output_idx = get_size_bucket_idx(output_length)

        # Get the weights relevant for this request size.
        weights = [backend["weights"][prompt_idx][output_idx] for backend in self.backends]

        rnd = random.random()
        cumulative_sum = 0
        for backend, weight in zip(self.backends, weights):
            cumulative_sum += weight
            if rnd <= cumulative_sum:
                return backend


# class WeightedRandom_InputOnly(LBPolicy):


# TODO:
# 1) JSQ routing - use %  load as queue. Should we drain the % load or just keep growing? establish some memory? 
# 2) Route based on ILP assignment