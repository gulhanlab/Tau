import pandas as pd
import numpy as np
import pysam
import argparse

def trim(sample, snv_file, cnv_file, purity_file, output_file=None):
    print(f"Processing sample: {sample}")
    print(f"SNV file: {snv_file}")
    print(f"CNV file: {cnv_file}")
    print(f"Purity file: {purity_file}")
    print(f"Output file: {output_file}")

    vcf = pysam.VariantFile(snv_file)

    cn = pd.read_csv(cnv_file, delimiter='\t')
    cn['chromosome'] = cn['chromosome'].astype(str).apply(lambda x: x if 'chr' not in x else x[3:])
    cn['minor_cn'] = cn['minor_cn'].fillna(-1).astype(int)
    cn['major_cn'] = cn['major_cn'].fillna(-1).astype(int)
    cn['segment_id'] = cn['chromosome'] + ':' + cn['start'].astype(str) + '-' + cn['end'].astype(str)

    if purity_file:
        df_purity = pd.read_csv(purity_file, delimiter='\t')
        sample_key = 'sample' if 'sample' in df_purity.columns else 'samplename'
        purity = df_purity[df_purity[sample_key] == sample]['purity'].values[0]
    else:
        purity = 0.5  # Default purity

    records = []
    for record in vcf:
        chrom = record.contig if 'chr' not in record.contig else record.contig[3:]
        pos = record.pos
        ref = record.ref
        alt = record.alts[0]
        info = record.info
        vaf = float(info['VAF']) / purity if 'VAF' in info else -1
        nalt = float(info['t_alt_count']) if 't_alt_count' in info else -1
        nref = float(info['t_ref_count']) if 't_ref_count' in info else -1
        true_mult = int(info['MUL']) if 'MUL' in info else -1

        records.append([chrom, vaf, pos, nalt, nref, alt, ref, purity, true_mult])

    df = pd.DataFrame(records, columns=['chrom', 'vaf', 'pos', 'nalt', 'nref', 'alt', 'ref', 'purity', 'true_multiplicity'])
    df['location'] = df['chrom'] + ':' + df['pos'].astype(str)

    df['major_cn'] = np.nan
    df['minor_cn'] = np.nan
    df['segment_id'] = np.nan
    df['segment_id'] = df['segment_id'].astype(str)

    for i, row in df.iterrows():
        chrom_eq = (cn['chromosome'] == row['chrom']).any()
        start_eq = (row['pos'] >= cn['start']).any()
        end_eq = (row['pos'] <= cn['end']).any()
        #print(f'chrom_eq: {chrom_eq}, start_eq: {start_eq}, end_eq: {end_eq}')
        #print(f'row position: {row["pos"]}, type: {type(row["pos"])}')
        this_cn = cn[(cn['chromosome'] == row['chrom']) & (row['pos'] >= cn['start']) & (row['pos'] <= cn['end'])]
        if not this_cn.empty:
            df.at[i, 'major_cn'] = this_cn['major_cn'].values[0]
            df.at[i, 'minor_cn'] = this_cn['minor_cn'].values[0]
            df.at[i, 'segment_id'] = this_cn['segment_id'].values[0]

    # Calculate total CN and adjusted VAF
    df['total_cn'] = df['minor_cn'] + df['major_cn']
    df['nref_corrected'] = np.round(df['nref'] - (1 - purity) * (df['nalt'] + df['nref']), 0)
    sd_nref_normal = np.sqrt((df['nref'] + df['nalt']) * (1 - purity) * purity)
    df['nref_corrected_low'] = np.round(df['nref_corrected'] - 2 * sd_nref_normal, 0)
    df['nref_corrected_high'] = np.round(df['nref_corrected'] + 2 * sd_nref_normal, 0)
    df['nref_corrected'] = df['nref_corrected'].clip(lower=0)
    df['nref_corrected_low'] = df['nref_corrected_low'].clip(lower=0)
    df['nref_corrected_high'] = df['nref_corrected_high'].clip(lower=0)
    df['vaf'] = df['nalt'] / (df['nalt'] + df['nref_corrected'])

    # Save the output
    if output_file:
        df.to_csv(output_file, sep='\t', index=False)
        print(f"Trim step completed successfully. Output saved to {output_file}")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim script for Tau workflow")
    parser.add_argument("sample", help="Sample name")
    parser.add_argument("snv_file", help="Path to the SNV file")
    parser.add_argument("cnv_file", help="Path to the CNV file")
    parser.add_argument("output_file", help="Path to the output file")
    parser.add_argument("purity_file", help="Path to the purity file")
    args = parser.parse_args()

    trim(
        sample=args.sample,
        snv_file=args.snv_file,
        cnv_file=args.cnv_file,
        purity_file=args.purity_file,
        output_file=args.output_file
    )
