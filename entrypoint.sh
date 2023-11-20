#!/bin/sh
if [ -z "${PREFIX}" ]; then
    PREFIX_PARAM="";
else
    PREFIX_PARAM="--prefix ${PREFIX}";
fi
bokeh serve --port ${PORT} --address 0.0.0.0 --log-level ${LOG_LEVEL} --show /InCongnitiveApp