import numpy as np
from typing import Optional, Tuple, List

# ============================================================================
# Core spectrum preparation functions for FFT analysis
# ============================================================================

def apodize(data: np.ndarray, n1: int, n2: int, percent: float) -> None:
    """
    Apodize the ends of a spectrum with a cosine bell.
    
    The cosine bell starts rising from zero at n1 and falls to zero at n2.
    
    Parameters:
    -----------
    data : np.ndarray
        Input spectrum data (modified in place)
    n1, n2 : int
        Start and end indices for apodization region
    percent : float
        Percentage of total data length to apodize
    """
    n = len(data)
    nsquash = int(min(n * 0.01 * percent, (n2 - n1) / 2))
    
    if nsquash < 1:
        return
    
    for i in range(nsquash):
        arg = np.pi * i / (nsquash - 1)
        factor = 0.5 * (1 - np.cos(arg))
        data[n1 + i] *= factor
        data[n2 - i] *= factor


def rebin(wave: np.ndarray, fsrc: np.ndarray, nlog: int, 
          w0: float, dwlog: float) -> np.ndarray:
    """
    Bin a spectrum in log wavelength.
    
    Parameters:
    -----------
    wave : np.ndarray
        Original wavelength array
    fsrc : np.ndarray
        Original flux array
    nlog : int
        Number of bins for log-rebinned spectrum
    w0 : float
        Reference wavelength for log scaling
    dwlog : float
        Logarithmic wavelength bin width
    
    Returns:
    --------
    np.ndarray : Rebinned flux array
    """
    nwave = len(wave)
    fdest = np.zeros(nlog)
    
    # Rebin each source pixel onto the destination array
    for l in range(nwave):
        # Give each source pixel boundaries half way between tabulated wavelengths
        if l == 0:
            s0 = 0.5 * (3 * wave[l] - wave[l + 1])
            s1 = 0.5 * (wave[l] + wave[l + 1])
        elif l == nwave - 1:
            s0 = 0.5 * (wave[l - 1] + wave[l])
            s1 = 0.5 * (3 * wave[l] - wave[l - 1])
        else:
            s0 = 0.5 * (wave[l - 1] + wave[l])
            s1 = 0.5 * (wave[l] + wave[l + 1])
        
        # Map boundaries to log wavelength array space
        s0log = np.log(s0 / w0) / dwlog + 1
        s1log = np.log(s1 / w0) / dwlog + 1
        
        # Using Flambda (as per Saurabh)
        dnu = s1 - s0
        
        # Run over rebinning loop
        i_start = int(np.floor(s0log))
        i_end = int(np.floor(s1log))
        
        for i in range(i_start, i_end + 1):
            if i < 0 or i >= nlog:
                continue
            
            alen = min(s1log, i + 1) - max(s0log, i)
            flux = fsrc[l] * alen / (s1log - s0log) * dnu
            fdest[i] += flux
    
    return fdest


def zerospec(nw: int) -> np.ndarray:
    """Initialize a flux array with zeros."""
    return np.zeros(nw)


def addspec(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """Add two flux arrays element-wise."""
    return f1 + f2
def mean_zero(y: np.ndarray, ioff: int = 0, nedge: int = 1) -> Tuple[np.ndarray, int, int, int, np.ndarray, np.ndarray]:
    """
    Normalize spectrum by dividing out spline continuum.
    
    Args:
        y: Input flux array
        ioff: Knot offset (>0 to stagger knots)
        nedge: Edge pixels to ignore
        
    Returns:
        Tuple of (ynorm, l1, l2, nknot, xknot, yknot)
    """
    n = len(y)
    KNOTNUM = 13  # Target knots per spectrum
    
    # Find valid data range, ignoring edges with zeros
    l1 = np.where(y > 0)[0][0] if np.any(y > 0) else 0
    l2 = np.where(y > 0)[0][-1] if np.any(y > 0) else n-1
    
    # Expand range to ignore edge zeros
    l1 = min(l1 + nedge, n-1)
    l2 = max(l2 - nedge, 0)
    
    if l2 - l1 < 3 * KNOTNUM:
        print('Warning: Spectrum has insufficient valid data')
        return y.copy(), l1, l2, 0, np.array([]), np.array([])
    
    # Choose knots using averaging method (default)
    kwidth = max(1, n // KNOTNUM)
    istart = (ioff % kwidth) - kwidth if ioff > 0 else 0
    
    # Collect knots from binned averages
    knots_x, knots_y = [], []
    for i in range(l1, l2, kwidth):
        chunk = y[i:min(i+kwidth, l2)]
        if np.all(chunk > 0):
            knots_x.append(i + len(chunk)/2)  # Center of bin
            knots_y.append(np.log10(np.mean(chunk)))
    
    if not knots_x:
        return y.copy(), l1, l2, 0, np.array([]), np.array([])
    
    xknot, yknot = np.array(knots_x), np.array(knots_y)
    nknot = len(xknot)
    
    # Create spline (or linear if few knots)
    if nknot < 3:
        spline = interp1d(xknot, yknot, kind='linear', fill_value='extrapolate')
    else:
        spline = CubicSpline(xknot, yknot, extrapolate=True)
    
    # Normalize: divide by spline and subtract 1
    ynorm = y.copy()
    indices = np.arange(l1, l2+1)
    continuum = 10.0 ** spline(indices - 0.5)
    ynorm[indices] = y[indices] / continuum - 1
    
    return ynorm, l1, l2, nknot, xknot, yknot

def meanzero(y: np.ndarray, ioff: int = 0, nedge: int = 1) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Fit a spline to the spectrum and normalize by dividing it out.
    
    Parameters:
    -----------
    y : np.ndarray
        Input flux array
    ioff : int
        Offset for picking knots if > 0
    nedge : int
        Number of edge pixels to ignore
    
    Returns:
    --------
    tuple: (ynorm, ynorm, l1, l2, nknot, xknot, yknot)
        Normalized array, modified normalized array, start/end indices,
        number of knots, knot x-positions, knot y-positions
    """
    import pylab as plt
    print("meanzero")
    n = len(y)
    KNOTCHOICE = 2  # 1 = maximum, 2 = average
    KNOTNUM = 13    # Normal number of knots over N pixels
    MAXKNOT = 20
    
    # Copy the array
    ynorm = y.copy()
    
    # Find the range of non-zero data values, ignoring edge pixels
    l1 = 0
    nuke = 0
    while l1 < n and (y[l1] <= 0 or nuke < nedge):
        if y[l1] > 0:
            nuke += 1
        ynorm[l1] = 0
        l1 += 1
    
    l2 = n - 1
    nuke = 0
    while l2 > 0 and (y[l2] <= 0 or nuke < nedge):
        if y[l2] > 0:
            nuke += 1
        ynorm[l2] = 0
        l2 -= 1
    
    if l2 - l1 < 3 * KNOTNUM:
        print('MEANZERO: This spectrum is zero!')
        return ynorm, l1, l2, 0, np.array([]), np.array([])
    
    # Choose knots based on selection method
    if KNOTCHOICE == 1:
        # Pick maximum values in bins
        nknot = max(5, (KNOTNUM * (l2 - l1)) // n)
        xknot = np.zeros(nknot)
        yknot = np.zeros(nknot)
        
        for k in range(nknot):
            if k == 0:
                i1 = l1
                i2 = l1
            elif k == nknot - 1:
                i1 = l2
                i2 = l2
            else:
                i1 = int((k - 1.5) * (l2 - l1 + 1) / (nknot - 1) + l1)
                i2 = int((k - 0.5) * (l2 - l1 + 1) / (nknot - 1) + l1 - 1)
            
            biggie = -500
            for i in range(i1, i2 + 1):
                if y[i] > biggie:
                    xknot[k] = i
                    yknot[k] = np.log10(y[i])
                    biggie = y[i]
    
    elif KNOTCHOICE == 2:
        # Use average values in bins
        kwidth = n // KNOTNUM
        istart = 0
        if ioff > 0:
            istart = (ioff % kwidth) - kwidth
        
        # Collect knot positions
        knots_x = []
        knots_y = []
        
        nave = 0
        wave_sum = 0
        fave_sum = 0
        istart = 0
        if ioff > 0:
            istart = (ioff % kwidth) - kwidth
        
        for i in range(n):
            if i > l1 and i < l2:
                nave += 1
                wave_sum += i - 0.5
                fave_sum += y[i]
            
            if (i - istart) % kwidth == 0:
                if nave > 0 and fave_sum > 0:
                    knots_x.append(wave_sum / nave)
                    knots_y.append(np.log10(fave_sum / nave))
                
                nave = 0
                wave_sum = 0
                fave_sum = 0
        
        nknot = len(knots_x)
        xknot = np.array(knots_x)
        yknot = np.array(knots_y)
    
    else:
        raise ValueError('MEANZERO: must make a choice for KNOTCHOICE')
    
    # Calculate spline (simplified - using cubic spline interpolation)
    if nknot < 4:
        # Fall back to linear interpolation for few points
        from scipy.interpolate import interp1d
        spline_interp = interp1d(xknot, yknot, kind='linear', 
                                fill_value='extrapolate')
    else:
        from scipy.interpolate import CubicSpline
        spline_interp = CubicSpline(xknot, yknot, extrapolate=True)
    
    # Evaluate the spline at each point, divide and subtract 1
    for i in range(l1, l2 + 1):
        spl = spline_interp(i - 0.5)
        #plt.plot(i, y[i], 'x')
        ynorm[i] = y[i] / (10.0 ** spl) - 1
        #plt.plot(i, ynorm[i], 'x')
    #plt.title("spline")
    #plt.show()
        
    return ynorm, l1, l2, nknot, xknot, yknot


def splinedex(n: int, loff: int, nknot: int, xknot: np.ndarray, 
             yknot: np.ndarray) -> np.ndarray:
    """
    Fill an array with spline evaluations.
    
    Parameters:
    -----------
    n : int
        Length of output array
    loff : int
        Offset for spline evaluation
    nknot : int
        Number of knots
    xknot, yknot : np.ndarray
        Knot positions and values
    
    Returns:
    --------
    np.ndarray : Array of spline evaluations
    """
    if nknot < 4:
        from scipy.interpolate import interp1d
        spline_interp = interp1d(xknot, yknot, kind='linear', 
                                fill_value='extrapolate')
    else:
        from scipy.interpolate import CubicSpline
        spline_interp = CubicSpline(xknot, yknot, extrapolate=True)
    
    y = np.zeros(n)
    for i in range(n):
        # Hold to constant offset of the region where the fit took place
        eval_idx = max(0, min(n - 1, i + loff)) - 0.5
        spl = spline_interp(eval_idx)
        y[i] = 10.0 ** spl
    
    return y


def overlap(x0: np.ndarray, y0: np.ndarray, shift: float, 
           percent: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate overlap and trim shifted buffers.
    
    Parameters:
    -----------
    x0, y0 : np.ndarray
        Input arrays (Y has SHIFT to the right wrt X)
    shift : float
        Shift between arrays
    percent : float
        Percentage for apodization
    
    Returns:
    --------
    tuple: (x1, y1, lap)
        Trimmed and apodized arrays, overlap length
    """
    n = len(x0)
    ishift = int(np.round(shift))
    
    # Find non-zero regions
    lx1 = np.argmax(x0 != 0)
    lx2 = n - 1 - np.argmax(x0[::-1] != 0)
    
    ly1 = np.argmax(y0 != 0)
    ly2 = n - 1 - np.argmax(y0[::-1] != 0)
    
    # Desired edges for overlap
    mx1 = max(lx1, ly1 - ishift)
    mx2 = min(lx2, ly2 - ishift)
    my1 = max(ly1, lx1 + ishift)
    my2 = min(ly2, lx2 + ishift)
    
    lap = mx2 - mx1 + 1
    
    # Copy and reapodize
    x1 = np.zeros_like(x0)
    y1 = np.zeros_like(y0)
    
    x1[mx1:mx2+1] = x0[mx1:mx2+1]
    y1[my1:my2+1] = y0[my1:my2+1]
    
    apodize(x1, mx1, mx2, percent)
    apodize(y1, my1, my2, percent)
    
    return x1, y1, lap


def xlog(x: float) -> float:
    """Return log10(x) with sign handling."""
    if x < 0:
        return -np.log10(-x)
    elif x == 0:
        return 0
    else:
        return np.log10(x)


def despace(s: str) -> str:
    """Remove leading spaces from a string."""
    return s.lstrip()


def medfilt(data: np.ndarray, medlen: int) -> np.ndarray:
    """
    Replace data with a running median.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    medlen : int
        Median filter length
    
    Returns:
    --------
    np.ndarray : Median-filtered data
    """
    n = len(data)
    buf = data.copy()
    medwidth = medlen // 2
    
    result = np.zeros_like(data)
    for k in range(n):
        i_start = max(0, k - medwidth)
        i_end = min(n, k + medwidth)
        result[k] = np.median(buf[i_start:i_end])
    
    return result


def medwfilt(wave: np.ndarray, data: np.ndarray, fwmed: float) -> np.ndarray:
    """
    Replace data with a weighted running median.
    
    Parameters:
    -----------
    wave : np.ndarray
        Wavelength array
    data : np.ndarray
        Flux array
    fwmed : float
        Full-width median smoothing width
    
    Returns:
    --------
    np.ndarray : Weighted median-filtered data
    """
    MAXDUP = 3
    n = len(data)
    buf = data.copy()
    
    # Gaussian width of our FWHM
    sig = fwmed / 2.35
    
    # Where are the count break points?
    brkpt = np.zeros(MAXDUP)
    for i in range(1, MAXDUP + 1):
        brkpt[i-1] = sig * np.sqrt(2 * np.log(MAXDUP / (i - 0.5)))
    
    result = np.zeros_like(data)
    
    for k in range(n):
        tmp = []
        for i in range(n):
            if abs(wave[i] - wave[k]) > brkpt[0]:
                continue
            
            for j in range(1, MAXDUP + 1):
                if abs(wave[i] - wave[k]) <= brkpt[j-1]:
                    tmp.append(buf[i])
                    if len(tmp) > n:
                        print(f'MEDWFILT: not enough array for this smoothing width {fwmed}')
                        return data
        
        if tmp:
            result[k] = np.median(tmp)
    
    return result


# ============================================================================
# Utility functions for completeness
# ============================================================================

def spline(xknot: np.ndarray, yknot: np.ndarray, nknot: int) -> Tuple[object, bool]:
    """
    Calculate cubic spline coefficients.
    
    Note: This is a simplified wrapper. In practice, use scipy's CubicSpline.
    """
    try:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(xknot[:nknot], yknot[:nknot])
        return cs, False
    except Exception as e:
        print(f'SPLINE: Error determining spline: {e}')
        return None, True


def splineval(x: float, cs: object) -> float:
    """Evaluate spline at point x."""
    if cs is None:
        return 0.0
    return cs(x)


def amedian(n: int, arr: np.ndarray) -> float:
    """Return median of first n elements of array."""
    if n <= 0:
        return 0.0
    return np.median(arr[:n])


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    # Example: Create a synthetic spectrum and process it
    n = 1000
    wave = np.linspace(4000, 8000, n)
    flux = 1000 + 500 * np.sin(wave/100) + np.random.normal(0, 50, n)
    
    # Test apodization
    flux_apo = flux.copy()
    apodize(flux_apo, 0, n-1, 5.0)
    
    # Test rebinning
    w0 = 5000.0
    dwlog = 0.01
    nlog = 500
    flux_rebin = rebin(wave, flux, nlog, w0, dwlog)
    
    # Test mean-zero normalization
    flux_norm, l1, l2, nknot, xknot, yknot = meanzero(flux)
    
    print("Processing complete!")
    print(f"Original flux shape: {flux.shape}")
    print(f"Rebinned flux shape: {flux_rebin.shape}")
    print(f"Normalization range: {l1} to {l2}")
    print(f"Number of knots: {nknot}")
