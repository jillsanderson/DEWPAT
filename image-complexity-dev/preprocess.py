""" Command line interface to visual_acuity.py
"""

import argparse, skimage, os
from csv import reader
from module.visual_acuity import acuity_view


def main():
    ### >> Argument parsing << ###
    parser = argparse.ArgumentParser(
                description='Acuity View',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument('input', type=str, help='Input: either a folder or an image')
    parser.add_argument('params', type=str,
                help='A CSV file containing the real width, viewing distance, and MRA for each image')
    parser.add_argument('output', type=str, help='Specifies output directory to which the blurred images are saved.')

    parser.add_argument('--verbose', dest='verbose', action='store_true',
                help='Whether to print verbosely while running')

    args = parser.parse_args()

    params = {}
    with open(args.params) as f:
        csv = reader(f, delimiter=',')
        next(csv, None)  # skip the header
        for row in csv:
            params[row[0]] = [float(n) for n in row[1:]]


    if os.path.isdir(args.input):
        usables = [ '.jpg', '.png' ]
        usables = list(set( usables + [ b.upper() for b in usables ] + 
                                      [ b.lower() for b in usables ] ))
        _checker = lambda k: any( k.endswith(yy) for yy in usables )
        targets = [ os.path.join(args.input, f) 
                    for f in os.listdir(args.input) if _checker(f) ]
        for t in targets: 
            main_helper(t, params, args)
    elif os.path.isfile(args.input):
        main_helper(args.input, params, args)
    else:
        raise ValueError('Non-existent target input ' + args.input)


def main_helper(img_path, params, args):
    img_file_basename = os.path.basename(img_path)
    img_file_basename_extless, _ = os.path.splitext(img_file_basename)

    if img_file_basename not in params:
        print(f'Parameters for {img_file_basename} could not be found in {args.params}')
        return    

    if args.verbose: print(f'Blurring {img_file_basename}')
    width, distance, MRA = params[img_file_basename]
    blurred_img = acuity_view(img_path, width, distance, R=MRA)

    # save the image
    if not os.path.isdir(args.output): os.makedirs(args.output)
    fname = os.path.join(args.output, img_file_basename_extless + '_blurred.png')
    skimage.io.imsave(fname=fname, arr=blurred_img)


if __name__ == '__main__':
    main()
