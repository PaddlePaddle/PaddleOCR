#!/usr/bin/env bash

if [ -z "$1" ]; then
    if [ -z "$FLASK_APP_PATH" ]; then
        echo "Environment Variable \`FLASK_APP_PATH\` should not be empty"
    elif [ -f "$FLASK_APP_PATH" ]; then
        python "$FLASK_APP_PATH"
    else
        echo "Environment Variable \`FLASK_APP_PATH\` is set to \"$FLASK_APP_PATH\" but it does not exist"
        exit
    fi
else
    bash -c "$1"
fi
