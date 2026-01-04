import numpy as np
import argparse
import glob
from scipy import fft, signal, interpolate

class SNID:
    """Supernova Identification - minimal Python implementation."""
    
    def __init__(self):
        # Core parameters
        self.nw = 1024  # Log wavelength bins
        self.w0, self.w1 = 2500, 10000  # Wavelength range
        self.dwlog = np.log(self.w1/self.w0)/self.nw
        self.lapmin = 0.4  # Minimum overlap
        self.rlapmin = 5.0  # Minimum rlap
        self.zfilter = 0.02  # Redshift filter
        
        # Data storage
        self.fnorm = None  # Input spectrum
        self.templates = []  # Template spectra
        self.names, self.types, self.epochs = [], [], []
        
    def read_spectrum(self, filename):
        """Read and normalize input spectrum."""
        # Parse wavelength/flux columns
        data = np.loadtxt(filename)
        wave, flux = data[:,0], data[:,1]
        
        # Log-rebin to common scale
        wlog = np.linspace(np.log(self.w0), np.log(self.w1), self.nw)
        self.fnorm = np.interp(wlog, np.log(wave), flux)
        
        # Normalize (remove continuum)
        self.fnorm -= np.polyval(np.polyfit(np.arange(self.nw), self.fnorm, 3), 
                                np.arange(self.nw))
        self.fnorm /= np.std(self.fnorm)
        
    def read_templates(self, template_dir):
        """Read all template .lnw files."""
        for fname in glob.glob(f"{template_dir}/*.lnw"):
            data = np.loadtxt(fname, skiprows=1)
            if data.shape[0] == self.nw:  # Check dimensions
                self.templates.append(data[:,1:])  # Multiple epochs
                self.names.append(fname.stem)
                # Extract type from filename/header (simplified)
                self.types.append(self._parse_type(fname))
                
    def correlate(self, spec1, spec2):
        """Cross-correlate two spectra."""
        # FFT-based correlation
        ft1 = fft.fft(spec1)
        ft2 = fft.fft(spec2)
        corr = fft.ifft(ft1 * ft2.conj()).real
        corr = np.roll(corr, len(corr)//2)  # Center zero-lag
        
        # Find peak
        peak_idx = np.argmax(corr)
        r = corr[peak_idx] / (np.std(spec1) * np.std(spec2) * len(spec1))
        z = (np.exp(peak_idx * self.dwlog) - 1)  # Convert to redshift
        
        # Calculate overlap
        non_zero = np.sum((spec1 != 0) & (spec2 != 0))
        lap = non_zero / len(spec1)
        
        return r, lap, z, corr
        
    def find_best_match(self):
        """Main SN identification routine."""
        best_rlap, best_z, best_name, best_type = 0, 0, "", ""
        results = []
        
        # First pass: initial redshift estimate
        all_peaks = []
        for i, template in enumerate(self.templates):
            for epoch in template.T:  # Iterate through epochs
                r, lap, z, _ = self.correlate(self.fnorm, epoch)
                rlap = r * lap
                all_peaks.append((rlap, z, self.names[i], self.types[i]))
                
        # Initial redshift from median of good peaks
        good_peaks = [z for rlap,z,_,_ in all_peaks if rlap > 4]
        z_init = np.median(good_peaks) if good_peaks else 0
        
        print(f"Initial redshift estimate: z = {z_init:.3f}")
        
        # Second pass: refined matching at initial redshift
        for i, template in enumerate(self.templates):
            # Find best epoch for this template
            best_epoch_rlap = 0
            for epoch in template.T:
                # Apply redshift shift (simplified)
                shift = int(np.log(z_init + 1) / self.dwlog)
                epoch_shifted = np.roll(epoch, shift)
                
                # Calculate overlap after shifting
                overlap_mask = (self.fnorm != 0) & (epoch_shifted != 0)
                if np.sum(overlap_mask)/len(epoch) < self.lapmin:
                    continue
                    
                r, lap, z, _ = self.correlate(self.fnorm[overlap_mask], 
                                            epoch_shifted[overlap_mask])
                rlap = r * lap
                
                if rlap > best_epoch_rlap:
                    best_epoch_rlap = rlap
                    best_epoch_z = z
            
            if best_epoch_rlap > self.rlapmin:
                results.append((best_epoch_rlap, best_epoch_z, 
                              self.names[i], self.types[i]))
                
        # Sort by rlap (descending)
        results.sort(reverse=True)
        
        # Type statistics
        if results:
            best_rlap, best_z, best_name, best_type = results[0]
            
            # Count type occurrences
            type_counts = {}
            for _, _, name, type_ in results[:10]:  # Top 10
                type_counts[type_] = type_counts.get(type_, 0) + 1
                
            print(f"\nBest match: {best_name} ({best_type})")
            print(f"Redshift: z = {best_z:.3f}, Rlap = {best_rlap:.1f}")
            print(f"\nType distribution in top matches:")
            for t, count in type_counts.items():
                print(f"  {t}: {count}")
                
        return results
        
    def _parse_type(self, filename):
        """Parse SN type from filename (simplified)."""
        # Real implementation would parse .lnw header
        name = filename.stem
        if 'Ia' in name: return 'Ia'
        if 'Ib' in name: return 'Ib' 
        if 'Ic' in name: return 'Ic'
        if 'II' in name: return 'II'
        return 'Unknown'
        
    def run(self, data_file, template_dir):
        """Main execution method."""
        print("SNID - Supernova Identification")
        print("Python minimal implementation\n")
        
        # Read data
        self.read_spectrum(data_file)
        print(f"Read input spectrum: {data_file}")
        
        # Read templates
        self.read_templates(template_dir)
        print(f"Read {len(self.templates)} templates from {template_dir}")
        
        # Run identification
        results = self.find_best_match()
        
        # Output summary
        print("\n" + "="*50)
        print("Top 5 matches:")
        for i, (rlap, z, name, type_) in enumerate(results[:5], 1):
            print(f"{i:2d}. {name:15s} {type_:5s} z={z:6.3f} rlap={rlap:5.1f}")
            
        return results

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNID - Supernova Identification")
    parser.add_argument("data", help="Input spectrum file")
    parser.add_argument("-t", "--templates", default="templates", 
                       help="Template directory")
    
    args = parser.parse_args()
    
    # Run SNID
    snid = SNID()
    snid.run(args.data, args.templates)
