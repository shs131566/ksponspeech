#!/usr/bin/env bash

data=$1

nj=16
vocab_size=100000

stage=1

source cmd.sh
source path.sh
. parse_options.sh

# name for data
train_dir=train
test_dir=test

# name for lm & subword data
subword_dir=subword
lm_dir=lm

# name for kaldi format
dict_nosp_dir=dict_nosp
lang_nosp_dir=lang_nosp
dict_dir=dict
lang_dir=lang
mfcc_dir=mfcc

set -e

if [ $stage -le 1 ]; then
    echo -e "Stage 1: Data preparation for ksponspeech"

    if [ ! -d data/local/$subword_dir ]; then
        mkdir -p data/local/$subword_dir
    fi

    if [ ! -d data/$train_dir ]; then
        mkdir -p data/$train_dir
    fi

    if [ ! -d data/$test_dir ]; then
        mkdir -p data/$test_dir/eval-clean
        mkdir -p data/$test_dir/eval-other
    fi

    # make wav.scp, text for train, eval-clean and eval-other.
    # sentences contain English are not used.
    python3 local/ksponspeech_preparation.py $data data/$train_dir \
        data/$test_dir/eval-clean data/$test_dir/eval-other

    # prepare text for subword and language model.
    # we use only train.trn of ksponspeech.
    cut -d ' ' -f2- data/$train_dir/text > data/local/$subword_dir/text

    # make utt2spk, spk2utt for train, eval-clean and eval-other.
    # there is no speaker information for ksponspeech.
    for dir in "data/$train_dir" "data/$test_dir/eval-clean" "data/$test_dir/eval-other"
    do
        awk ' {print $1, $1 }' $dir/text > $dir/utt2spk
        utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
    done
    utils/validate_data_dir.sh --no-feats data/$train_dir
    utils/validate_data_dir.sh --no-feats data/$test_dir/eval-clean
    utils/validate_data_dir.sh --no-feats data/$test_dir/eval-other
fi

if [ $stage -le 2 ]; then
    echo -e "Stage 2: Train subword tokenizer"

    spm_train --input data/local/$subword_dir/text \
        --model_prefix=data/local/$subword_dir/subword \
        --model_type=bpe --hard_vocab_limit=false --vocab_size=$vocab_size \
        --character_coverage=1.0
fi

if [ $stage -le 3 ]; then
    echo -e "Stage 3: Tokenize text corpus for train data."

    if [ ! -d data/local/$lm_dir ]; then
        mkdir -p data/local/$lm_dir
    fi

    spm_encode --model data/local/$subword_dir/subword.model \
        --output data/local/$lm_dir/text < data/local/$subword_dir/text

    echo "spm_encode: `wc -l data/local/$subword_dir/text` lines are used."
fi

if [ $stage -le 4 ]; then
    echo -e "Stage 4: Train n-gram language model."

    norder=2
    prune_prob=1e-7

    # make language model vocab from subword vocab.
    cut -f1 data/local/$subword_dir/subword.vocab | sed '/^_/d' > \
        data/local/$lm_dir/vocab

    # train n-gram language model.
    ngram-count -vocab data/local/$lm_dir/vocab -text data/local/$lm_dir/text \
        -order $norder -lm data/local/$lm_dir/lm.arpa -prune $prune_prob -debug 2
fi

if [ $stage -le 5 ]; then
    echo -e "Stage 5: Make lexicon from g2p"
    # make lexicon by G2P.
    # we use KoG2P (https://github.com/scarletcho/KoG2P)
    sed -n '4, $ p' data/local/$lm_dir/vocab | python3 local/g2p/g2p.py \
        | sed '/^▁ /d' | sort -u > data/local/$lm_dir/lexicon_nosil.txt

    echo "local/g2p/g2p.py: `wc -l data/local/$lm_dir/lexicon_nosil.txt` \
        words are used"
fi

if [ $stage -le 6 ]; then
    echo -e "Stage 6: Format the data as KALDI data directories"

    # make dict_nosp, lang_nosp
    local/prepare_dict.sh data/local/$lm_dir data/local/$dict_nosp_dir
    utils/prepare_lang.sh data/local/$dict_nosp_dir "<UNK>" \
        data/local/$lang_nosp_dir data/$lang_nosp_dir

    # Make G.fst
    cat data/local/$lm_dir/lm.arpa | arpa2fst --disambig-symbol=#0 \
        --read-symbol-table=data/$lang_nosp_dir/words.txt \
        - data/$lang_nosp_dir/G.fst
    utils/validate_lang.pl --skip-determinization-check data/$lang_nosp_dir
fi

if [ $stage -le 7 ]; then
    echo -e "Stage 7: Feature extraction"
    # extract MFCC feature from data.
    # see options of MFCC --> conf/mfcc.conf

    # feature extraction for train data.
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/$train_dir \
        exp/make_mfcc/$train_dir $mfcc_dir/$train_dir
    steps/compute_cmvn_stats.sh data/$train_dir exp/make_mfcc/$train_dir \
        $mfcc_dir/$train_dir

    # feature extraction for test data
    for dir in "eval-clean" "eval-other"; do
        steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj \
            data/$test_dir/$dir exp/make_mfcc/$test_dir/$dir
        steps/compute_cmvn_stats.sh data/$test_dir/$dir \
            exp/make_mfcc/$test_dir/$dir $mfcc_dir/$test_dir/$dir
    done
fi

if [ $stage -le 8 ]; then
    echo -e "Stage 8: Mono phone training & alignment"

    # we only use small data set for mono phone train.
    utils/subset_data_dir.sh data/$train_dir 10000 data/${train_dir}_10K
    steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
        data/${train_dir} data/$lang_nosp_dir exp/mono

    # TODO: it needs to be modified for full data alignments.
    steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
        data/${train_dir} data/$lang_nosp_dir exp/mono exp/mono_ali
fi

if [ $stage -le 9 ]; then
    echo -e "Stage 9: tri phone training [tri1] & alignment"

    npdf=2000
    ngaussian=10000
    steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" $npdf \
        $ngaussian data/${train_dir} data/$lang_nosp_dir exp/mono_ali exp/tri1
    steps/align_si.sh --nj $nj --cmd "$train_cmd" data/${train_dir} \
        data/$lang_nosp_dir exp/tri1 exp/tri1_ali
fi

if [ $stage -le 10 ]; then
    echo -e "Stage 10: tri phone training [tri2b] & alignment"

    npdf=2500
    ngaussian=15000

    steps/train_lda_mllt.sh --cmd "$train_cmd" --splice_opts \
        "--left-context=3 --right-context=3" $npdf $ngaussian \
        data/${train_dir} data/$lang_nosp_dir exp/tri1_ali exp/tri2b
    steps/align_si.sh --nj $nj --cmd "$train_cmd" data/${train_dir} \
        data/$lang_nosp_dir exp/tri2b exp/tri2b_ali
fi

if [ $stage -le 11 ]; then
    echo -e "Stage 11: tri phone training [tri3b] & alignment"

    npdf=2500
    ngaussian=15000

    steps/train_sat.sh --cmd "$train_cmd" $npdf $ngaussian \
        data/${train_dir} data/$lang_nosp_dir exp/tri2b_ali exp/tri3b
    steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" data/${train_dir} \
        data/$lang_nosp_dir exp/tri3b exp/tri3b_ali
fi

if [ $stage -le 12 ]; then
    echo -e "Stage 12: tri phone training [tri4b] & alignment"

    npdf=4200
    ngaussian=40000

    steps/train_sat.sh --cmd "$train_cmd" $npdf $ngaussian data/${train_dir} \
        data/$lang_nosp_dir exp/tri3b_ali exp/tri4b

    # add silence probability
    # modify dict/lang folder calculateed silence probabilty from {lang/dict}_nosp
    steps/get_prons.sh --cmd "$train_cmd" data/${train_dir} \
        data/$lang_nosp_dir exp/tri4b
    utils/dict_dir_add_pronprobs.sh --max-normalize true \
        data/local/$dict_nosp_dir exp/tri4b/pron_counts_nowb.txt \
        exp/tri4b/sil_counts_nowb.txt exp/tri4b/pron_bigram_counts_nowb.txt \
        data/local/$dict_dir
    utils/prepare_lang.sh data/local/$dict_dir "<UNK>" data/local/$lang_nosp_dir \
        data/$lang_dir

    # update silence probability to G.fst
    cat data/local/$lm_dir/lm.arpa | arpa2fst --disambig-symbol=#0 \
        --read-symbol-table=data/$lang_dir/words.txt - data/$lang_dir/G.fst

    utils/validate_lang.pl --skip-determinization-check data/$lang_dir

    steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" data/${train_dir} \
        data/$lang_dir exp/tri4b exp/tri4b_ali
fi

if [ $stage -le 13 ]; then
    echo -e "Stage 13: tri phone training [tri5b] & alignments"

    npdf=5000
    ngaussian=100000

    steps/train_sat.sh --cmd "$train_cmd" $npdf $ngaussian \
        data/${train_dir} data/$lang_dir exp/tri4b_ali exp/tri5b

    steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" data/${train_dir} \
        data/$lang_dir exp/tri5b exp/tri5b_ali
fi

if [ $stage -le 14 ]; then
    echo -e "Stage 14: tri phone training [tri6b] & alignments"

    npdf=70000
    ngaussian=150000

    steps/train_quick.sh --cmd "$train_cmd" $npdf $ngaussian \
        data/${train_dir} data/$lang_dir exp/tri5b_ali exp/tri6b


    steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" data/${train_dir} \
        data/$lang_dir exp/tri6b exp/tri6b_ali
fi

if [ $stage -le 15 ]; then
    echo -e "Stage 15: make decoding network by tri6b model"

    utils/mkgraph.sh data/$lang_dir exp/tri6b exp/tri6b/grpah
fi

if [ $stage -le 16 ]; then
    echo -e "Stage 16: preparing speed-perturbation."

    utils/fix_data_dir.sh data/${train_dir}_10K
    utils/data/perturb_data_dir_speed_3way.sh data/${train_dir}_10K data/${train_dir}_sp

    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/${train_dir}_sp
    steps/compute_cmvn_stats.sh data/${train_dir}_sp

    steps/align_fmllr.sh data/${train_dir}_sp data/$lang_dir exp/tri6b exp/tri6b_ali_${train_dir}_sp
fi

if [ $stage -le 17 ]; then
    echo -e "Stage 17: preparing volume-perturbation."

    utils/copy_data_dir.sh data/${train_dir}_sp data/${train_dir}_sp_hires
    utils/data/perturb_data_dir_volume.sh data/${train_dir}_sp_hires
fi

if [ $stage -le 18 ]; then
    echo -e "Stage 18: make high-resolution MFCC for perturbation data."

    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc_config conf/mfcc_hires.conf \
        data/${train_dir}_sp_hires
    steps/compute_cmvn_stats.sh data/${train_dir}_sp_hires
    utils/fix_data_dir.sh data/${train_dir}_sp_hires
fi

if [ $stage -le 19 ]; then
    echo -e "Stage 19: TDNN train."

    local/nnet3/run_tdnn.sh
fi
