import mrcfile
import pickle
import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib as mpl
import argparse

def add_args(parser):
    parser.add_argument('--mask_projections', type = str, required = True, help = '.mrcs file with projected mask images')
    parser.add_argument('--particles', type = str, required = True, help = 'particle stack files, either .mrcs or .txt')
    parser.add_argument('--outfile', type = str, required = True, help = 'name of output .mrcs with masked particle images')
    parser.add_argument('--ind', type = str, default = None, help = 'optionally provide a set of indices to filter the particle image stack by')
    return parser
    
def main(args):
    # first read in the mask projections
    with mrcfile.open(args.mask_projections, 'r', permissive=True) as mrc:
        mask = mrc.data.copy()
    
    
    # identify the .mrcs particle image file(s)
    assert args.particles.endswith('.txt') or args.particles.endswith('.mrcs'), 'particles must either be .mrcs or .txt file!'
    if args.particles.endswith('.txt'):
        with open(args.particles, 'r') as f:
            particles = f.read().split('\n')
    elif args.particles.endswith('.mrcs'):
        particles = [args.particles]
    
    
    # read in indices to filter particle stack by, if appropriate
    if args.ind is not None:
        with open(args.ind, 'rb') as f:
            filter_inds = pickle.load(f)
    else:
        filter_inds = np.arange(len(mask))
    
    
    # initialize empty masked particle array
    boxsize = mask.shape[1]
    masked_particles = np.zeros((len(filter_inds), boxsize, boxsize), dtype = 'float32')
    
    
    # map across multiple chunks if necessary
    with mrcfile.open(particles[0], 'r', permissive = True) as mrc:
        chunksize, __, __ = mrc.data.shape
    index_chunk_mapping = filter_inds//chunksize
    
    
    # iterate over each .mrcs file
    for j, m in enumerate(particles):
    
        # read in particle images
        with mrcfile.open(m, 'r', permissive = True) as mrc:
            im = mrc.data.copy()
    
        # find the right particle and multiply it by its corresponding mask projection
        for i in np.where(index_chunk_mapping == j):
            fi = filter_inds[i]-j*chunksize
            masked_particles[i] = mask[i]*im[fi]


    # save a figure of the first mask projection + masked particle image
    fig, ax = plt.subplots(1, 2, figsize = (6, 4))
    ax = ax.flatten()
    
    ind = 0
    
    ax[0].imshow(mask[ind])
    ax[1].imshow(masked_particles[ind], cmap = 'gray')
    
    if not args.outfile.endswith('.mrcs'):
        args.outfile = f'{args.outfile}.mrcs'
    fig.savefig(f'{args.outfile.split(".mrcs")[0]}_masked_particle_0.png')

    # write out .mrcs with masked particle images
    with mrcfile.new(args.outfile, overwrite=True) as output_mrc:                     
        output_mrc.set_data(masked_particles)
        output_mrc.update_header_from_data()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(add_args(parser).parse_args())
