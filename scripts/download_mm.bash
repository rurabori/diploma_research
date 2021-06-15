#! /usr/bin/env bash

pushd resources/matrices
curl -L $1 --output o.tar.gz
tar xvfz o.tar.gz
rm o.tar.gz
popd
