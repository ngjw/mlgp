#!/bin/bash

demodir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" 
cd "$demodir"; ./demo.out
