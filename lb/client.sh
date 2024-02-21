# curl -X POST -H "Content-Type: application/json" -d '{"key":"value"}' http://example.com/api -o /dev/null -w "%{redirect_url}"

# curl -X POST -H "Content-Type: application/json" -d '{"key":"value"}' [REDIRECT_URL]


# curl -L http://0.0.0.0:8000/forward \
#     -X POST \
#     -d '{"model": "meta-llama/Llama-2-7b-hf", "prompt": "Hello, world", "max_tokens": 100}' \
#     -H 'Content-Type: application/json'

#!/bin/bash

# Initial URL and JSON data
initial_url="http://0.0.0.0:8000/forward"
json_data='{"model": "meta-llama/Llama-2-7b-hf", "prompt": "Hello, world", "max_tokens": 10}'

# Make the initial POST request and capture the redirect URL
redirect_url=$(curl -X POST -H "Content-Type: application/json" -d "$json_data" -s -o /dev/null -w "%{redirect_url}" "$initial_url")

# Check if we got a redirect URL
if [ -n "$redirect_url" ]; then
    echo "Redirect URL found: $redirect_url"
    # Make a second POST request to the redirect URL with the same JSON data
    curl -X POST -H "Content-Type: application/json" -d "$json_data" "$redirect_url"
else
    echo "No redirect URL found."
fi

echo "\n"
