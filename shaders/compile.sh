#!/bin/bash


pushd shaders

glslc -fshader-stage=vert -o vert.spv vertshader.glsl
glslc -fshader-stage=frag -o frag.spv fragshader.glsl
