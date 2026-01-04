import numpy as np
import sys
import os
from typing import List, Tuple, Dict
import argparse
from apodize import meanzero, apodize
# ============================================================================
# Configuration (translated from snid.inc)
# ============================================================================

import numpy as np
from typing import Tuple
from scipy.interpolate import CubicSpline, interp1d

class SNIDConfig:
    """Configuration parameters for SNID/Logwave (from snid.inc)."""
    
    # Constants
    EPSFRAC = 5e-4
    EPSRLAP = 5e-2
    EPSSLOPE = 5e-4
    EPSZ = 5e-4
    MAXEPOCH = 300
    MAXKNOT = 20
    MAXLIST = 20
    MAXLOG = 1024
    MAXPARAM = 200
    MAXPEAK = 20
    MAXPLOT = 20
    MAXPPT = 50_000
    MAXR = 999.9
    MAXRLAP = 999
    MAXSN = 300
    MAXUSE = 30
    MAXTEMP = 10_000
    MAXTOK = 32
    MAXWAVE = 10_000
    MINWAVE = 2_500.0
    NT = 5      # Number of primary types
    NST = 6     # Maximum subtypes + 1
    
    def __init__(self):
        # User options
        self.w0 = 2_500     # Default wavelength range
        self.w1 = 10_000
        self.nw = 1024          # Log wavelength bins
        self.percent = 5.0      # Apodization percentage
        
        # Working arrays
        self.wave = np.zeros(self.nw)
        self.flux = np.zeros(self.nw)
        
    

# ============================================================================
# Spectrum Processing Functions
# ============================================================================

def log_rebin(wave: np.ndarray, flux: np.ndarray, nw: int, 
              w0: float, w1: float) -> np.ndarray:
    """Rebin spectrum to log wavelength scale."""
    # Create log wavelength grid
    log_wave = np.logspace(np.log10(w0), np.log10(w1), nw)
    # Interpolate to new grid
    return np.interp(log_wave, wave, flux)

'''

def apodize(data: np.ndarray, percent: float = 5.0) -> np.ndarray:
    """Apply cosine bell apodization to ends."""
    n = len(data)
    n_apo = int(n * percent / 100)
    if n_apo > 0:
        # Cosine bell
        i = np.arange(n_apo)
        window = 0.5 * (1 - np.cos(np.pi * i / (n_apo - 1)))
        data[:n_apo] *= window
        data[-n_apo:] *= window[::-1]
    return data
'''
import numpy as np
from typing import Tuple
from scipy.interpolate import CubicSpline, interp1d



#mean_zero = meanzero
# ============================================================================
# Main Logwave Processor
# ============================================================================

class Logwave:
    """Convert spectra to common log wavelength scale."""
    
    def __init__(self, config: SNIDConfig = None):
        self.config = config or SNIDConfig()
        
    def read_spectrum(self, filename: str, #ab_flag: int, 
                      wmin: float, wmax: float#, redshift: float) -> Tuple[np.ndarray, np.ndarray]:
                      ):
        """Read wavelength and flux from file."""
        try:
            data = np.loadtxt(filename)
            wave, flux = data[:, 0], data[:, 1]
            
            ## Apply AB magnitude conversion if needed
            #if ab_flag == 1:
            #    flux = 10.0 ** (-0.4 * flux)
            
            # Apply wavelength mask
            mask = (wave >= wmin) & (wave <= wmax)
            wave, flux = wave[mask], flux[mask]
            
            ## Apply redshift correction
            #if redshift > 0:
            #    wave = wave / (1 + redshift)
            
            return wave, flux
        except:
            print(f"Error reading {filename}")
            return np.array([]), np.array([])
    
    def process_spectrum(self, filename, wmin, wmax): #-> Dict:
        """Process all epochs for a supernova."""
        from apodize import meanzero as mean_zero

        # Initialize arrays
        nw = self.config.nw
        flog = np.zeros(nw)
        fnorm = np.zeros(nw)
        fmean = [0]
        nknot = np.zeros(1, dtype=int)
        
        # Setup log wavelength grid
        dwlog = np.log(self.config.w1 / self.config.w0) / nw
        wlog = self.config.w0 * np.exp(np.arange(nw + 1) * dwlog)
        
        # Process each epoch
        most_knots = 0
        # Read spectrum
        wave, flux = self.read_spectrum(
            filename,  wmin, wmax)
        
        
        assert len(wave) > 0, "no data"
            
        
        # Rebin to log scale
        flog = log_rebin(wave, flux, nw, self.config.w0, self.config.w1)
        
        # Normalize (remove continuum)
        fnorm, l1, l2, nknot, xknot, yknot = mean_zero(flog)
        #print("here", fnorm)
        most_knots = max(most_knots, nknot)
        
        # Apodize
        #fnorm = apodize(fnorm, 10,
        #                      -10, self.config.percent)
        
        # Calculate mean flux for scaling
        fmean = np.mean(flog)
        flog = flog / fmean
        
        # Prepare output
        output = {
            'wlog': wlog,
            'flog': flog,
            'fnorm': fnorm,
            'fmean': fmean,
            'nknot': nknot
        }
        
        return output
    
    def process_supernova(self, spectra: List[Dict]) -> Dict:
        """Process all epochs for a supernova."""
        if not spectra:
            return None
        
        #sn_name = spectra[0]['object']
        #n_epochs = len(spectra)
        
        # Initialize arrays
        nw = self.config.nw
        flog = np.zeros((nw, n_epochs))
        fnorm = np.zeros((nw, n_epochs))
        fmean = np.zeros(n_epochs)
        nknot = np.zeros(n_epochs, dtype=int)
        
        # Setup log wavelength grid
        dwlog = np.log(self.config.w1 / self.config.w0) / nw
        wlog = self.config.w0 * np.exp(np.arange(nw + 1) * dwlog)
        
        # Process each epoch
        most_knots = 0
        for j, spec in enumerate(spectra):
            # Read spectrum
            wave, flux = self.read_spectrum(
                spec['filename'], spec['ab_flag'], 
                spec['wmin'], spec['wmax'], spec['redshift']
            )
            
            if len(wave) == 0:
                continue
            
            # Rebin to log scale
            flog[:, j] = log_rebin(wave, flux, nw, self.config.w0, self.config.w1)
            
            # Normalize (remove continuum)
            fnorm[:, j], nknot[j] = mean_zero(flog[:, j])
            most_knots = max(most_knots, nknot[j])
            
            # Apodize
            fnorm[:, j] = apodize(fnorm[:, j], 10,
                                 -10, self.config.percent)
            
            # Calculate mean flux for scaling
            mask = flog[:, j] > 0
            if np.any(mask):
                fmean[j] = np.mean(flog[mask, j])
                flog[:, j] = flog[:, j] / fmean[j]
        
        # Prepare output
        output = {
            'name': sn_name,
            'type': spectra[0]['type'],
            'delta': spectra[0]['delta'],
            'n_epochs': n_epochs,
            'nw': nw,
            'w0': self.config.w0,
            'w1': self.config.w1,
            'most_knots': most_knots,
            'wlog': wlog,
            'flog': flog,
            'fnorm': fnorm,
            'fmean': fmean,
            'nknot': nknot,
            'ages': [s['age'] for s in spectra],
            'age_flags': [s['age_flag'] for s in spectra]
        }
        
        return output
    
    def write_output(self, output: Dict, filename: str):
        """Write processed data to .lnw file."""
        with open(filename, 'w') as f:
            # Header line
            f.write(f"{output['n_epochs']:5d}{output['nw']:5d}"
                   f"{output['w0']:10.2f}{output['w1']:10.2f}"
                   f"{output['most_knots']:7d}     {output['name']:12s}"
                   f"{output['delta']:7.2f}  {output['type']:10s}  1  1\n")
            
            # Knot information
            f.write(f"{output['most_knots']:7d}")
            for j in range(output['n_epochs']):
                f.write(f"{output['nknot'][j]:3d}{np.log10(output['fmean'][j]):14.5f}")
            f.write("\n")
            
            # Placeholder for knot positions (simplified)
            for i in range(output['most_knots']):
                f.write(f"{i+1:7d}")
                for j in range(output['n_epochs']):
                    f.write("        0.0000  0.0000")
                f.write("\n")
            
            # Ages
            f.write(f"{output['age_flags'][0]:8d}")
            for age in output['ages']:
                f.write(f"{age:9.3f}")
            f.write("\n")
            
            # Wavelength and normalized flux
            for i in range(output['nw']):
                wmean = 0.5 * (output['wlog'][i] + output['wlog'][i+1])
                f.write(f"{wmean:8.2f}")
                for j in range(output['n_epochs']):
                    f.write(f"{output['fnorm'][i, j]:9.3f}")
                f.write("\n")
    
    def run(self, list_file: str, w0: float = None, w1: float = None, nw: int = None):
        """Main processing routine."""
        print("Logwave - Spectrum preparation for SNID")
        
        # Update config if provided
        if w0 is not None:
            self.config.w0 = w0
        if w1 is not None:
            self.config.w1 = w1
        if nw is not None:
            self.config.nw = nw
        
        # Check parameters
        if self.config.nw > self.config.MAXLOG:
            print(f"Error: nw={self.config.nw} exceeds MAXLOG={self.config.MAXLOG}")
            return
        
        # Parse list file
        print(f"Reading list from: {list_file}")
        all_spectra = self.parse_list_file(list_file)
        
        if not all_spectra:
            print("No valid spectra found in list file")
            return
        
        # Group by supernova
        sn_spectra = {}
        for spec in all_spectra:
            obj = spec['object']
            if obj not in sn_spectra:
                sn_spectra[obj] = []
            sn_spectra[obj].append(spec)
        
        # Process each supernova
        for sn_name, spectra in sn_spectra.items():
            print(f"Processing {sn_name} ({len(spectra)} epochs)...")
            
            # Process supernova
            output = self.process_supernova(spectra)
            if output is None:
                print(f"  Failed to process {sn_name}")
                continue
            
            # Write output
            out_file = f"{sn_name}.lnw"
            self.write_output(output, out_file)
            print(f"  Created: {out_file}")
        
        print("\nProcessing complete!")

# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Convert spectra to common log wavelength scale for SNID"
    )
    parser.add_argument("list_file", help="List of spectra to process")
    parser.add_argument("--w0", type=float, default=2500.0,
                       help="Minimum wavelength (default: 2500)")
    parser.add_argument("--w1", type=float, default=10000.0,
                       help="Maximum wavelength (default: 10000)")
    parser.add_argument("--nw", type=int, default=1024,
                       help="Number of log wavelength bins (default: 1024)")
    
    args = parser.parse_args()
    
    # Run Logwave
    logwave = Logwave()
    logwave.run(args.list_file, args.w0, args.w1, args.nw)

if __name__ == "__main__":
    main()
