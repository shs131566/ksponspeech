#!/bin/bash

#    This is the standard "tdnn" system, built in nnet3 with xconfigs.

set -e -o pipefail -u

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
nj=10

train_set=train_sp_hires
test_sets=test
gmm=tri6b        # this is the source gmm-dir that we'll use for alignments; it
                 # should have alignments for the specified training data.
num_threads_ubm=1
nnet3_affix=       # affix for exp dirs, e.g. it was _cleaned in tedlium.
tdnn_affix=6b
#affix for TDNN directory e.g. "1a" or "1b", in case we change the configuration.

# Options which are not passed through to run_ivector_common.sh
train_stage=-10
remove_egs=false
srand=0
reporting_email=
# set common_egs_dir to use previously dumped egs.
common_egs_dir=

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

#if ! cuda-compiled; then
  #cat <<EOF && exit 1
#This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
#If you want to use GPUs (and have them), go to src/, and configure and make on a machine
#where "nvcc" is installed.
#EOF
#fi


gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_train_sp
dir=exp/nnet3/tdnn${tdnn_affix}
train_data_dir=data/${train_set}
train_ivector_dir=" "

#for f in $train_data_dir/feats.scp \
#    $gmm_dir/graph/HCLG.fst \
#    $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
#  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
#done


if [ $stage -le 12 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $gmm_dir/tree | grep num-pdfs | awk '{print $2}')

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=1280
  relu-renorm-layer name=tdnn2 dim=1280 input=Append(-1,0,1)
  relu-renorm-layer name=tdnn3 dim=1280 input=Append(-1,0,1)
  relu-renorm-layer name=tdnn4 dim=1280 input=Append(-1,0,1)
  relu-renorm-layer name=tdnn5 dim=1280 input=Append(-3,0,3)
  relu-renorm-layer name=tdnn6 dim=1280 input=Append(-6,-3,0)
  output-layer name=output dim=$num_targets max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

echo "check..."

if [ $stage -le 13 ]; then

  steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=4 \
    --trainer.optimization.num-jobs-initial=4 \
    --trainer.optimization.num-jobs-final=4 \
    --trainer.optimization.initial-effective-lrate=0.0017 \
    --trainer.optimization.final-effective-lrate=0.00017 \
    --trainer.optimization.minibatch-size=256,128 \
    --egs.dir="$common_egs_dir" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=false \
    --feat-dir=$train_data_dir \
    --ali-dir=$ali_dir \
    --lang=data/lang \
    --dir=$dir  || exit 1;
fi


if [ $stage -eq 14 ]; then
  # note: for TDNNs, looped decoding gives exactly the same results
  # as regular decoding, so there is no point in testing it separately.
  # We use regular decoding because it supports multi-threaded (we just
  # didn't create the binary for that, for looped decoding, so far).
  rm $dir/.error || true 2>/dev/null
  for data in $test_sets; do
      data_affix=$(echo $data | sed s/test_//)
      nj=4
      graph_dir=$gmm_dir/graph
      steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd"  --num-threads 1 \
        ${graph_dir} data/${data} ${dir}/decode_${data_affix}

      steps/lmrescore.sh --cmd "$decode_cmd" data/lang \
        data/${data} ${dir}/decode_${data_affix}

     # steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
     #   data/lang data/${data}_hires ${dir}/decode_${data_affix}
  done
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi


exit 0;
