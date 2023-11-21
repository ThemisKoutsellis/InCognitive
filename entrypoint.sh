#!/bin/sh
bokeh serve --port 5006 --address 0.0.0.0 --allow-websocket-origin=${ORIGIN} --show /InCognitiveApp