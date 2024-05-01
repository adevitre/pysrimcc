'''
    pysrimcc
    Calculations of radiation damage (ion particle range, dpa, impurity concentration, and heat) in coated-conductors
    @author Alexis Devitre (devitre@mit.edu)

    References used for this work
    [Gray 2022] R L Gray et al 2022 Supercond. Sci. Technol. 35 035010
    [Konobeyev 2017] Nuclear Energy and Technology 3(3):169
    [Short 2013] Journal of Nuclear Materials 471
    [Ziegler 2010] Nuclear Instruments and Methods in Physics Research Section B, Volume 268, Issue 11-12, p. 1818-1823.
    
'''

import os, time, json, srim, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from srim import TRIM, Ion, Layer, Target
from srim.output import Results, Range
from datetime import datetime

# displacement threshold energies for YBCO from [Gray 2022] and, for the metal layers, we can take values from [Konobeyev 2017]; In lack of a better reference, we take the YBCO values and metal values to be true for constituents of the ceramic buffers.

Ed_O1 = np.mean([9.141, 12.95, 15.11, 17.43, 19.92, 15.11])
Ed_O2 = np.mean([12.95, 25.39, 12.95, 25.39, 15.11, 67.34])
Ed_O3 = np.mean([15.11, 12.95, 12.95, 17.43, 25.39, 17.43])
Ed_O4 = np.mean([15.11, 17.43, 38.32, 15.11, 2.5, 22.57])
Ed_Cu1 = np.mean([9.961, 9.961, 23.79, 13.91, 6.668])
Ed_Cu2 = np.mean([9.961, 29.72, 29.72, 36.31, 9.961, 13.91])

Ed_O = np.mean([Ed_O1, Ed_O2, Ed_O3, Ed_O4])
Ed_Cu = np.mean([Ed_Cu1, Ed_Cu2])
Ed_Ba = np.mean([78.46, 30.07, 51.42, 4.45, 51.42, 30.07])
Ed_Y = np.mean([33.287, 33.287, 50.794, 33.287, 41.58, 41.58])

Ed_Ag = 39
Ed_Ni = 33
Ed_Mo = 65
Ed_Fe = 40
Ed_Cr = 40
Ed_W = 90

Ed_La = 29
Ed_Mn = 33
Ed_Mg = 20
Ed_Al = 27

composition = {
    'ybco': {
        'elements': {
            'Y': {'stoich': 1.0, 'E_d': Ed_Y, 'lattice': Ed_Y},
            'Ba': {'stoich': 2.0, 'E_d': Ed_Ba, 'lattice': Ed_Ba},
            'Cu': {'stoich': 3.0, 'E_d': Ed_Cu, 'lattice': Ed_Cu},
            'O': {'stoich': 7.0, 'E_d': Ed_O, 'lattice': Ed_O}
        },
        'density': 6.3
    },
    'ag': {
        'elements': {
            'Ag': {'stoich': 1.0, 'E_d': Ed_Ag, 'lattice': Ed_Ag}
        },
        'density': 10.473
    },
    'lamno3': {
        'elements': {
            'La': {'stoich': 1.0, 'E_d': Ed_La, 'lattice': Ed_La},
            'Mn': {'stoich': 1.0, 'E_d': Ed_Mn, 'lattice': Ed_Mn},
            'O': {'stoich': 3.0, 'E_d': Ed_O, 'lattice': Ed_O}
        },
        'density': 6.54
    },
    'mgo': {
        'elements': {
            'Mg': {'stoich': 1.0, 'E_d': Ed_Mg, 'lattice': Ed_Mg},
            'O': {'stoich': 1.0, 'E_d': Ed_O, 'lattice': Ed_O}
        },
        'density': 3.58
    },
    'y2o3': {
        'elements': {
            'Y': {'stoich': 2.0, 'E_d': Ed_Y, 'lattice': Ed_Y},
            'O': {'stoich': 3.0, 'E_d': Ed_O, 'lattice': Ed_O}
        },
        'density': 5.01
    },
    'al2o3': {
        'elements': {
            'Al': {'stoich': 2.0, 'E_d': Ed_Al, 'lattice': Ed_Al},
            'O': {'stoich': 3.0, 'E_d': Ed_O, 'lattice': Ed_O}
        },
        'density': 3.95
    },
    'c-270': {
        'elements': {
            'Ni': {'stoich': 55.0, 'E_d': Ed_Ni, 'lattice': Ed_Ni},
            'Mo': {'stoich': 17.0, 'E_d': Ed_Mo, 'lattice': Ed_Mo},
            'Cr': {'stoich': 16.0, 'E_d': Ed_Cr, 'lattice': Ed_Cr},
            'Fe': {'stoich': 7.0, 'E_d': Ed_Fe, 'lattice': Ed_Fe},
            'W': {'stoich': 4.5, 'E_d': Ed_W, 'lattice': Ed_W}
        },
        'density': 8.89
    }
}

SRIM_EX_DIR = 'C://Program Files (x86)/SRIM'
SRIM_OUT_DIR = 'C:/Program Files (x86)/SRIM'

def setUserNameSpace(srim_ex_dir, srim_out_dir):
    SRIM_EX_DIR = srim_ex_dir
    SRIM_OUT_DIR = srim_out_dir
    
def loadTapeArchitecture(fpath):
    tape = None
    with open(fpath, 'r') as f:
        tape = json.load(f)
    return tape
    
def maketarget(layers):
    tapeStack, layerBoundaries, layerTypes = [], [0], ['vacuum']
    for i, layer in enumerate(list(layers.items())):
        tapeStack.append(Layer(**composition[layer[0]], width=layer[1]))
        layerBoundaries.append(layer[1]+layerBoundaries[i])
        layerTypes.append(layer[0])
    return Target(tapeStack), layerBoundaries, layerTypes

def runSRIM(species, energy, ionsFlown, path_to_tape, calculation=1, savedir=None, vb=False):
    t0 = time.time()
    idxrun, cut, number = 0, 50000, ionsFlown
    if savedir is not None:
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        savedir = savedir+'/outdir_{}_{}_{}_{}_{}'.format(str(datetime.now()).replace(' ', '_').replace(':', '-'), species, energy, ionsFlown, calculation)
        os.mkdir(savedir)
    ranges, frenkelPairs, implantedIons, phononHeat, ionizationHeat = [], [], [], [], []

    # We run 50k ions at a time and reset the random seed on every loop see [Short 2013].
    # This process stops when the number of ions remaining is 1 or less. PySRIM won´t run a single ion simulation.
    while number > 1:
        target, bounds, layers = maketarget(layers=loadTapeArchitecture(path_to_tape))
        trim = TRIM(target, Ion(species, energy), number_ions=np.min([number, cut]), calculation=calculation)
        results = trim.run(SRIM_EX_DIR)
        
        ranges = results.ioniz.depth
        frenkelPairs.append((np.array(results.vacancy.knock_ons) + np.array([v[0] for v in results.vacancy.vacancies])))
        implantedIons.append(results.range.ions)
        phononHeat.append(results.phonons.ions + results.phonons.recoils)
        ionizationHeat.append(results.ioniz.ions + results.ioniz.recoils)
    
        if idxrun == 0:
            if ionsFlown < cut:
                t = time.time()-t0
                if vb: print('Completed {:5.0f} ions in {:6.0f} seconds.'.format(ionsFlown, t))
            else:
                t = time.time()-t0
                if vb:
                    print('Completed {:5.0f} ions in {:6.0f} seconds.'.format(cut, t))
                    print('This run will take a total of {:6.1f} minutes.'.format(t*ionsFlown/cut/60))
        if savedir is not None:
            savedirpart = savedir+'/part{}'.format(idxrun)
            os.mkdir(savedirpart) # This prevents overwriting the SRIM file copied over in the last 50k ion loop
            TRIM.copy_output_files(SRIM_OUT_DIR, savedirpart)
        idxrun += 1
        number -= cut
        if vb: print('Time elapsed {:4.2f} min, Ions flown {:5.0f}/{:5.0f}'.format((time.time()-t0)/60, ionsFlown-number, ionsFlown))
        
    frenkelPairs = np.mean(frenkelPairs, axis=0)
    implantedIons = np.mean(implantedIons, axis=0)
    phononHeat = np.mean(phononHeat, axis=0)
    ionizationHeat = np.mean(ionizationHeat, axis=0)
        
    if savedir is not None:
        cnames = ['depth_A', 'phonons_eV-Aion', 'ionizations_eV-Aion', 'frenkelpairs_vac-Aion', 'implantedIons_-cm']
        datestring = str(datetime.now()).replace(':', '-').replace('.', '-').replace(' ', '-')
        with open('{}/{}-{}{}-{}keV-{}.txt'.format(savedir,  datestring, ionsFlown, species, int(energy/1e3), calculation), 'w') as f:
            f.write('{:<25}\t{:<25}\t{:<25}\t{:<25}\t{:<25}\n'.format(*cnames))
            for (r, ph, ih, fp, ii) in zip(ranges, phononHeat, ionizationHeat, frenkelPairs, implantedIons):
                f.write('{:<25.10e}\t{:<25.10e}\t{:<25.10e}\t{:<25.10e}\t{:<25.10e}\n'.format(r, ph, ih, fp, ii))
            f.close()
    if vb: print('Hurray! We are finished. This run took {} seconds.'.format(str(time.time()-t0)))
    results = {
        'ranges': ranges,
        'heat': phononHeat+ionizationHeat,
        'frenkelPairs': frenkelPairs,
        'implantedIons': implantedIons
    }
    return np.array(bounds), pd.DataFrame(results)

def getIonRange(implantedIons, ranges):
    for i, ii in enumerate(implantedIons[::-1]):
        if ii != 0:
            break
    return ranges[len(implantedIons)-i]
    
def plotResults(bounds, data, species, energy, nions, tape):
    r = getIonRange(data.implantedIons, data.ranges)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    fig.suptitle('{} {}-ions with {} keV in {} tape; range = {:4.2f} um'.format(nions, species, energy/1e3, tape.upper(), r/1e4), fontsize=18)
    
    for ax, y, l in zip(axes, list(data.items())[1:], ['Heat profile [eV/A-ion]', 'Damage profile [vac/A-ion]', 'Ion implantation profile [ions/A-ion]']):
        xmax = y[1].max()
        for b0, b1, c in zip(bounds[:-1]/1e4, bounds[1:]/1e4, sns.color_palette('icefire', n_colors=len(bounds))):
            ax.fill_between([0, xmax], b0, b1, color=c, alpha=.3)
        ax.plot(y[1], data.ranges/1e4, color='k', linewidth=2, marker='+')
        ax.axhline(r/1e4, linestyle='--', linewidth=1, color='k')
        ax.set_xlabel(l, fontsize=12)
        ax.set_ylabel('Depth', fontsize=12)
        ax.set_ylim(0, 40)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
    return fig, axes