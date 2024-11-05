#! /bin/bash

mkdir -p result
log_file=result/bench.csv

echo "algo,n,m,v,mean,median,std,len" > $log_file

python bench_end2end_dense.py -bs 16 >> $log_file

###############
cd ../end2end
./install_v64.sh
cd ../sddmm_module
./install_v64.sh
cd ../deliverable
###############
python bench_end2end.py -m 8 -v 64 -bs 16 >> $log_file

python bench_end2end.py -m 16 -v 64 -bs 16 >> $log_file

python bench_end2end.py -m 32 -v 64 -bs 16 >> $log_file


###############
cd ../end2end
./install.sh
cd ../sddmm_module
./install.sh
cd ../deliverable
###############
python bench_end2end.py -m 8 -v 128 -bs 16 >> $log_file

python bench_end2end.py -m 16 -v 128 -bs 16 >> $log_file

python bench_end2end.py -m 32 -v 128 -bs 16 >> $log_file