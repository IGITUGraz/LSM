#!/bin/bash
CHECKER="pep8 --max-line-length 120"
find . -name "*.py" -exec $CHECKER {} \;
