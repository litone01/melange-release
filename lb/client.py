import aiohttp
import asyncio
import csv
import random
import time
from datetime import datetime
import sys
from typing import AsyncGenerator, List, Tuple
import numpy as np

# Stats:
# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []
# (prompt len, output len, [token latencies])
TOKEN_LATENCY: List[Tuple[int, int, List[float]]] = []
TIME_TO_FIRST_TOKEN: List[float] = []


# Randomly sample from dataset
def get_requests(dataset_file, num_requests):
  data = []
  with open(dataset_file, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(tuple(row))

  request_lens = random.sample(data, num_requests)

  requests = []
  for inlen, outlen in request_lens:
     inlen = int(inlen)
     outlen = int(outlen)
     requests.append(("hi " * inlen, inlen, outlen))
  
  return requests

async def send_request(prompt, prompt_len, output_len):
  api_url = 'http://0.0.0.0:8000/forward'

  headers = {
      'Content-Type': 'application/json'
  }

  pload = {
      "prompt": prompt,
      "n": 1,
      "best_of": 1,
      "use_beam_search": False,
      "temperature": 0.0,
      "top_p": 1.0,
      "max_tokens": output_len,
      "ignore_eos": True,
      "stream": True,
      "model": "meta-llama/Llama-2-7b-hf",
      # "messages": [{"role": "user", "content": prompt}],
  }

  try:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3 * 3600)) as session:
      backend_url = None
      # Query LB until 
      while True:
        async with session.post(api_url, headers=headers, json=pload, allow_redirects=False) as response:
          # Check for redirect.
          if response.status == 302:
            # Extract the location header
            location = response.headers.get('Location')
            if location:
                backend_url = location
                break
          print(f'[{datetime.now()}] Did not get backend URL from load balancer. Trying again.')
          time.sleep(0.1)
      print(f"[{datetime.now()}] Redirect URL: {location}")
    
      request_start_time = time.perf_counter()
      while True:
        async with session.post(backend_url, headers=headers, json=pload, allow_redirects=False) as response:
          chunks = []
          token_latencies = []
          previous_token_time = time.perf_counter()
          first = True
          async for chunk, _ in response.content.iter_chunks():
            # Stream on: Each chunk in the response is the full response so far
            chunks = [chunk]

            now_time = time.perf_counter()
            if first:
                time_to_first = now_time - previous_token_time
                first = False
            else:
                token_latencies.append(now_time - previous_token_time)
            previous_token_time = now_time

            # Stream off: Chunks are full response.
            # chunks.append(chunk)
        break
    print('Request completed.')
    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))
    TOKEN_LATENCY.append((prompt_len, output_len, token_latencies))
    TIME_TO_FIRST_TOKEN.append(time_to_first)

  except Exception as e:
    # Handle other exceptions, such as network errors
    print(f"[{datetime.now()}] An error occurred: {e}")
      
async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)

async def benchmark(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []

    # exp_start_time = time.time()
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(send_request(prompt, prompt_len, output_len))
        tasks.append(task)

        # Run the experiment for 3 minutes (or until all prompts are served).
        # if time.time() - exp_start_time > 180:
        #     break
    await asyncio.gather(*tasks)

def run_experiment(exp_name, dataset_file, request_rate, num_requests):
  benchmark_start_time = time.perf_counter()
  requests = get_requests(dataset_file, num_requests)
  asyncio.run(benchmark(requests, request_rate))
  benchmark_end_time = time.perf_counter()
  benchmark_time = benchmark_end_time - benchmark_start_time
  print()
  print("RESULT SUMMARY")
  print(f"Request rate: {request_rate} req/s")
  print(f"Prompt count: {len(REQUEST_LATENCY)}")
  print(f"Total time: {benchmark_time:.2f} s")
  print(f"Request Throughput: {len(REQUEST_LATENCY) / benchmark_time:.2f} requests/s")
  print(f"Output Token Throughput: {sum([output for _, output, _ in REQUEST_LATENCY]) / benchmark_time:.2f} tokens/s")
  print()

  now = datetime.now()
  timestamp_str = now.strftime('%Y%m%d_%H%M%S')

  # Write per-request latency to CSV file
  with open(f'csv/{exp_name}_{timestamp_str}.csv', 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerows(REQUEST_LATENCY)

  # Compute the latency statistics.
  avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
  print("REQUEST LATENCIES")
  print(f"Avg: {avg_latency:.2f} s")
  print(f"50p: {np.percentile([latency for _, _, latency in REQUEST_LATENCY], 50)} s")
  print(f"90p: {np.percentile([latency for _, _, latency in REQUEST_LATENCY], 90)} s")
  print(f"99p: {np.percentile([latency for _, _, latency in REQUEST_LATENCY], 99)} s")
  print()

  print('Easy Paste:')
  # reqrate,numPrompts,totalTime,reqThroughput, outputTokenThroughput, 50p req lat, 99p req lat
  print(f'=SPLIT("{request_rate},{len(REQUEST_LATENCY)},{benchmark_time:.2f},{len(REQUEST_LATENCY) / benchmark_time},{sum([output for _, output, _ in REQUEST_LATENCY]) / benchmark_time},{np.percentile([latency for _, _, latency in REQUEST_LATENCY], 50)},{np.percentile([latency for _, _, latency in REQUEST_LATENCY], 90)},{np.percentile([latency for _, _, latency in REQUEST_LATENCY], 99)}", ",")')


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print('Bad usage')
    #     exit(1)
    # port = sys.argv[1]

    exp_name = 'testing-exp'
    dataset_file = 'datasets/lmsys-chat-1m.csv'
    request_rate = 10
    num_requests = 100

    run_experiment(exp_name, dataset_file, request_rate, num_requests)

#     [[0.01 0.04 0.01 0.02 0.02 0.01]
#  [1.59 3.02 0.86 1.08 1.67 0.98]
#  [5.79 5.38 1.99 3.79 4.03 3.02]
#  [5.67 4.17 3.92 4.45 4.2  2.93]
#  [6.57 6.28 4.02 3.5  2.81 1.87]
#  [5.99 3.97 1.95 1.74 1.11 1.52]]