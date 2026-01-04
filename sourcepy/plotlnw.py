import numpy as np
import matplotlib.pyplot as plt
import sys

def plotlnw(filename, **kwargs):
    """Plot .lnw files from Logwave."""
    ips = kwargs.get('ps', 0)
    k1, k2, k3, k4 = kwargs.get('k1', 1), kwargs.get('k2', 4), kwargs.get('k3', 0), kwargs.get('k4', 0)
    yoff = kwargs.get('yoff', 1.0)
    
    # Read file
    with open(filename) as f:
        header = f.readline().split()
        nepoch, nw = int(header[0]), int(header[1])
        w0, w1 = float(header[2]), float(header[3])
        tname, dta, ttype = header[6], float(header[7]), header[8]
        
        # Skip knots
        for _ in range(1 + int(header[4])):
            f.readline()
        
        # Read epochs
        ep = list(map(float, f.readline().split()[1:1+nepoch]))
        
        # Read spectra
        wave, temp = [], []
        for _ in range(nw):
            parts = list(map(float, f.readline().split()))
            wave.append(parts[0])
            temp.append(parts[1:1+nepoch])
    
    wave, temp = np.array(wave), np.array(temp).T
    
    # Filter if requested
    if kwargs.get('filter', 0):
        k3 = k3 or nw//12
        k4 = k4 or nw//10
        for i in range(nepoch):
            ft = np.fft.fft(temp[i])
            ft[:k1] = ft[k2:k3] = ft[k4:] = 0
            temp[i] = np.fft.ifft(ft).real
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(nepoch):
        ax.plot(wave, temp[i] + (nepoch-i-1)*yoff, lw=1)
        ax.text(0.02, (nepoch-i-0.65)/(nepoch+1), f'{ep[i]:+.0f}', 
                transform=ax.transAxes, va='center')
    
    ax.set(xlim=(kwargs.get('xmin', w0), kwargs.get('xmax', w1)), 
           xlabel='Rest Wavelength [Å]', ylabel='Flattened Flux',
           title=f'SNID template {tname} ({ttype}) ; {nepoch} spectra')
    ax.set_yticks([])
    
    if ips:
        plt.savefig(f'{filename[:-4]}.ps')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: plotlnw template.lnw [options]')
        sys.exit()
    
    # Parse options
    kwargs = {}
    for arg in sys.argv[2:]:
        if '=' in arg:
            k, v = arg.split('=')
            kwargs[k] = int(v) if v.isdigit() or (v[0]=='-' and v[1:].isdigit()) else float(v)
    
    plotlnw(sys.argv[1], **kwargs)
