#!/usr/bin/env sh
/home/es1video4/caffe/build/tools/caffe train --solver=MRCNN_solver.prototxt --snapshot=./result_seg/eucilidean_0/data/model_iter_3000000.solverstate 2>&1 | tee ./result_seg/eucilidean_1/output.txt
