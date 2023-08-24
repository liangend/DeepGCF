'''
This script samples the first base of every non-overlapping genomic window defined across consecutive human genomic
bases that align to pig, converting aligning pairs of human and pig bases into aligning pairs of human and pig
regions.
The main output file (ending with .gz) has 18 columns:
- Columns 1-4 (chrom, start, end, nucleotide) correspond to the first human base of the 50-bp window
- Columns 5-8 correspond to the aligning pig base of the start of the 50-bp block
- Column 9 is the pair index of the start of the 50-bp block
- Columns 10-13 correspond to the human of the end of the 50-bp block
- Columns 14-17 correspond to the aligning pig base of the end of the 50-bp block
- Coummn 18 is the pair index of the end of the 50-bp block
We also generate separate output files for human regions (file ending with .h.gz) and pig regions (file ending  with
.m.gz)
'''

import gzip,argparse,os

def main():
    description = 'Sample the first base of every non-overlapping genomic window defined across consecutive human \
    genomic bases that align to pig'
    epilog = '# Example: python src/samplePairs.py -i position/hg38.susScr11.basepair.gz -o position/hg38.susScr11.50bp'

    parser = argparse.ArgumentParser(prog='python src/samplePairs.py',
                                     description=description,
                                     epilog=epilog)
    g1 = parser.add_argument_group('required arguments')
    g1.add_argument('-i','--input-filename',
                    help='path to output filename from aggregateAligningBases containing aligning pairs of human and pig bases',
                    required=True,type=str)
    parser.add_argument('-b','--bin-size',
                    help='size (bp) of the non-overlapping genomic window (default: 50)',
                    type=int,default=50)
    g1.add_argument('-o','--output-prefix',
                    help='prefix for output files',
                    required=True,type=str)
    args = parser.parse_args()

    with gzip.open(args.input_filename) as fin, gzip.open(args.output_prefix+'.gz','wb') as fout:

        # Write the very first line
        line = fin.readline().strip()
        fout.write(line+b'\t')
        l = line.split()
        hg_chrom = l[0].decode('utf-8')
        hg_pos = int(l[1])
        pig_pos = int(l[5])

        i = 1
        prev_chrom = hg_chrom
        prev_pos = hg_pos
        prev_line = line
        prev_pig = pig_pos

        for line in fin: # Iterate
            line = line.strip()
            l = line.split()
            hg_chrom = l[0].decode('utf-8')
            hg_pos = int(l[1])
            pig_pos = int(l[5])

            # Current base is contiguous to the previous base
            if prev_chrom == hg_chrom and hg_pos == prev_pos + 1 and abs(pig_pos - prev_pig) == 1:
                if i%args.bin_size==0:
                    fout.write(prev_line+b'\n')
                    fout.write(line+b'\t')
                i += 1

            # Current base is a start of new block
            else:
                fout.write(prev_line+b'\n')
                fout.write(line+b'\t')
                i = 1

            # Store current data
            prev_chrom = hg_chrom
            prev_pos = hg_pos
            prev_line = line
            prev_pig = pig_pos

        fout.write(line)

    # Write separate files for human and pig regions
    os.system("gzip -cd %s.gz | awk -v OFS='\t' '{print $1,$2,$3,$9}' | gzip > %s.h.gz" % (args.output_prefix,args.output_prefix))
    os.system("gzip -cd %s.gz | awk -v OFS='\t' '{print $5,$6,$7,$9}' | gzip > %s.m.gz" % (args.output_prefix,args.output_prefix))

main()
