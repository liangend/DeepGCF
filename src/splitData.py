import gzip,argparse

class LoadFromFile (argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            contents = f.read()

        data = parser.parse_args(contents.split(), namespace=namespace)
        for k, v in vars(data).items():
            if v and k != option_string.lstrip('-'):
                setattr(namespace, k, v)

def main():
    description = 'Split input data for training, validation, test, and model comparison based on human and pig chromosomes'
    epilog = '# Example: python src/splitData.py \
    -A position/hg38.susScr11.50bp.h.gz -B position/hg38.susScr11.50bp.m.gz -N 40268928 -o data/ @example/splitData_args.txt'

    parser = argparse.ArgumentParser(prog='python src/splitData.py', description=description, epilog=epilog)

    g1 = parser.add_argument_group('required arguments specifying input and output')

    g1.add_argument('-A','--human-region-filename',type=str,required=True,
                    help='path to combined human region filename')
    g1.add_argument('-B','--pig-region-filename',type=str,required=True,
                    help='path to combined mouse region filename')
    g1.add_argument('-N','--total-num-pairs',type=int,required=True,
                    help='total number of pairs')
    g1.add_argument('-o','--output-dir',type=str,required=True)

    g2 = parser.add_argument_group('required arguments specifying the split')

    g2.add_argument('--odd-training-human-chrom',required=True,
                    help='human chromosomes to include in odd training data',type=str,nargs='+')
    g2.add_argument('--odd-validation-human-chrom',required=True,
                    help='human chromosomes to include in odd validation data',type=str,nargs='+')
    g2.add_argument('--odd-test-human-chrom',required=True,
                    help='human chromosomes to include in odd test data', type=str, nargs='+')
    g2.add_argument('--odd-prediction-human-chrom',required=True,
                    help='human chromosomes to include in odd prediction data',type=str,nargs='+')

    g2.add_argument('--odd-training-pig-chrom',required=True,
                    help='pig chromosomes to include in odd training data',type=str,nargs='+')
    g2.add_argument('--odd-validation-pig-chrom',required=True,
                    help='pig chromosomes to include in odd validation data',type=str,nargs='+')
    g2.add_argument('--odd-test-pig-chrom',required=True,
                    help='pig chromosomes to include in odd test data', type=str, nargs='+')

    g2.add_argument('--even-training-human-chrom',required=True,
                    help='human chromosomes to include in even training data',type=str,nargs='+')
    g2.add_argument('--even-validation-human-chrom',required=True,
                    help='human chromosomes to include in even validation data',type=str,nargs='+')
    g2.add_argument('--even-test-human-chrom',required=True,
                    help='human chromosomes to include in even test data',type=str,nargs='+')
    g2.add_argument('--even-prediction-human-chrom',required=True,
                    help='human chromosomes to include in even prediction data',type=str,nargs='+')

    g2.add_argument('--even-training-pig-chrom',required=True,
                    help='pig chromosomes to include in even training data',type=str,nargs='+')
    g2.add_argument('--even-validation-pig-chrom',required=True,
                    help='pig chromosomes to include in even validation data',type=str,nargs='+')
    g2.add_argument('--even-test-pig-chrom',required=True,
                    help='pig chromosomes to include in even test data',type=str,nargs='+')

    args = parser.parse_args()

    # Convert lists of chromosomes into sets
    odd_training_human_chrom = set(args.odd_training_human_chrom)
    odd_validation_human_chrom = set(args.odd_validation_human_chrom)
    odd_test_human_chrom = set(args.odd_test_human_chrom)
    odd_prediction_human_chrom = set(args.odd_prediction_human_chrom)

    odd_training_pig_chrom = set(args.odd_training_pig_chrom)
    odd_validation_pig_chrom = set(args.odd_validation_pig_chrom)
    odd_test_pig_chrom = set(args.odd_test_pig_chrom)

    even_training_human_chrom = set(args.even_training_human_chrom)
    even_validation_human_chrom = set(args.even_validation_human_chrom)
    even_test_human_chrom = set(args.even_test_human_chrom)
    even_prediction_human_chrom = set(args.even_prediction_human_chrom)

    even_training_pig_chrom = set(args.even_training_pig_chrom)
    even_validation_pig_chrom = set(args.even_validation_pig_chrom)
    even_test_pig_chrom = set(args.even_test_pig_chrom)

    ### Start reading and writing
    hin = gzip.open(args.human_region_filename)
    min = gzip.open(args.pig_region_filename)

    with gzip.open(args.output_dir+'/odd_training.h.gz','wb') as human_odd_training_out,\
        gzip.open(args.output_dir+'/odd_validation.h.gz','wb') as human_odd_validation_out, \
        gzip.open(args.output_dir+'/odd_test.h.gz','wb') as human_odd_test_out, \
        gzip.open(args.output_dir+'/odd_all.h.gz','wb') as human_odd_prediction_out,\
        gzip.open(args.output_dir+'/even_training.h.gz','wb') as human_even_training_out,\
        gzip.open(args.output_dir+'/even_validation.h.gz','wb') as human_even_validation_out, \
        gzip.open(args.output_dir+'/even_test.h.gz','wb') as human_even_test_out, \
        gzip.open(args.output_dir+'/even_all.h.gz', 'wb') as human_even_prediction_out, \
        gzip.open(args.output_dir+'/odd_training.m.gz','wb') as pig_odd_training_out,\
        gzip.open(args.output_dir+'/odd_validation.m.gz','wb') as pig_odd_validation_out, \
        gzip.open(args.output_dir+'/odd_test.m.gz','wb') as pig_odd_test_out, \
        gzip.open(args.output_dir+'/odd_all.m.gz','wb') as pig_odd_prediction_out,\
        gzip.open(args.output_dir+'/even_training.m.gz','wb') as pig_even_training_out,\
        gzip.open(args.output_dir+'/even_validation.m.gz','wb') as pig_even_validation_out, \
        gzip.open(args.output_dir+'/even_test.m.gz','wb') as pig_even_test_out, \
        gzip.open(args.output_dir+'/even_all.m.gz', 'wb') as pig_even_prediction_out:

        for i in range(args.total_num_pairs):
            hline = hin.readline()
            mline = min.readline()
            hl = hline.strip().split()
            ml = mline.strip().split()
            hg_chrom = hl[0].decode('utf-8')
            mm_chrom = ml[0].decode('utf-8')

            # Write lines if the current human chromosome is one of the specified chromosomes
            if hg_chrom in odd_prediction_human_chrom:
                human_odd_prediction_out.write(hline)
                pig_odd_prediction_out.write(mline)

                if hg_chrom in odd_training_human_chrom and mm_chrom in odd_training_pig_chrom:
                    human_odd_training_out.write(hline)
                    pig_odd_training_out.write(mline)

                if hg_chrom in odd_validation_human_chrom and mm_chrom in odd_validation_pig_chrom:
                    human_odd_validation_out.write(hline)
                    pig_odd_validation_out.write(mline)

                if hg_chrom in odd_test_human_chrom and mm_chrom in odd_test_pig_chrom:
                    human_odd_test_out.write(hline)
                    pig_odd_test_out.write(mline)

            if hg_chrom in even_prediction_human_chrom:
                human_even_prediction_out.write(hline)
                pig_even_prediction_out.write(mline)

                if hg_chrom in even_training_human_chrom and mm_chrom in even_training_pig_chrom:
                    human_even_training_out.write(hline)
                    pig_even_training_out.write(mline)

                if hg_chrom in even_validation_human_chrom and mm_chrom in even_validation_pig_chrom:
                    human_even_validation_out.write(hline)
                    pig_even_validation_out.write(mline)

                if hg_chrom in even_test_human_chrom and mm_chrom in even_test_pig_chrom:
                    human_even_test_out.write(hline)
                    pig_even_test_out.write(mline)

main()
