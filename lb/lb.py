from flask import Flask, request, redirect, jsonify
from datetime import datetime
import sys
import lb_policies
import logging

now = datetime.now()
timestamp_str = now.strftime('%Y%m%d_%H%M%S')

logging.basicConfig(filename=f'logs/lb_{timestamp_str}.log', level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')
app = Flask(__name__)

lb_policy = lb_policies.WeightedRandom_InputAndOutput()
# lb_policy = lb_policies.RoundRobin()

LB_STATS = []

# We don't need this for now because we can just populate a 
# prompt with words that are a single token each (e.g., 'hi')
WORD_TO_TOKEN_MULTIPLIER = 1

@app.route('/v1/completions', methods=['POST'])
def completions():
    app.logger.info(f'Got request: {request.json}')

    data = request.json
    prompt = data.get('prompt', '')

    return jsonify({'message': 'Success'})


@app.route('/forward', methods=['POST'])
def forward():
    app.logger.info(f'Got request: {request.json}')

    data = request.json
    prompt = data.get('prompt', '')
    
    prompt_length = len(prompt.split()) * WORD_TO_TOKEN_MULTIPLIER
    app.logger.info(f'Prompt is length {prompt_length}')
    
    output_length = None
    if 'max_tokens' in data:
        output_length = data['max_tokens']

    backend = lb_policy.LoadBalance(prompt_length, output_length)
    (ip, port) = backend["url"]

    app.logger.info(f'({prompt_length}, {output_length}) --> ({backend["gpu_type"]}, {backend["url"]})')

    # Collect stats on the LB decision.
    LB_STATS.append((f'{datetime.now()}', prompt_length, output_length, backend["gpu_type"], backend["url"]))

    redirect_url = f'http://{ip}:{port}/v1/completions'

    return redirect(redirect_url, code=302)
    

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Bad usage')
        exit(1)
    port = sys.argv[1]

    # lb_policy.SetBackends([('34.173.38.88', 8000)])
    # AWS A10G: (3.137.200.198, 8000)
    # GCP A100: (34.28.94.90, 8000)
    lb_policy.SetBackends(
        [
            {"gpu_type" : "A10G", "url" : ('3.137.200.198', 8000)},
            {"gpu_type" : "A100", "url" : ('34.28.94.90', 8000)},
        ]
    )

    lb_policy.ComputeWeights()

    try:
        app.run(debug=False, host='0.0.0.0', port=port)
    finally:
        app.logger.info(f'LOAD BALANCER STATS: \n{LB_STATS}')

        # TODO: do some stats analysis here to dump to log

        for handler in app.logger.handlers:
            handler.flush()

    

