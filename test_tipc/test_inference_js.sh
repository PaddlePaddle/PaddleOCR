#!/bin/bash

set -o errexit
set -o nounset

cd test_tipc/web
# run ocr test in chrome
./node_modules/.bin/jest --config ./jest.config.js
