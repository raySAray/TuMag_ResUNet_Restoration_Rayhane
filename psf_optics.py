import math
import numpy as np
import pandas as pd

def noll_to_nm(j):
    """Convert Noll index j (j >= 1) to radial degree n and azimuthal frequency m."""
    if j < 1:
        raise ValueError("Noll index j must be >= 1")
    n = 0
    j1_val = j - 1
    while j1_val > n:
        n += 1
        j1_val -= n
    m = (-1)**j * ((n % 2) + 2 * int((j1_val + ((n + 1) % 2)) / 2))
    return n, m

def zernike_radial(n, m, rho):
    """Compute radial Zernike polynomial R_n^|m|(rho)."""
    m = abs(m)
    R = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        num = (-1)**k * math.factorial(n - k)
        den = (math.factorial(k) * 
               math.factorial((n + m) // 2 - k) * 
               math.factorial((n - m) // 2 - k))
        R += (num / den) * (rho**(n - 2 * k))
    return R

def zernike_noll(j, rho, theta):
    """Compute Noll-normalized Zernike polynomial Z_j(rho, theta) on unit disk."""
    n, m = noll_to_nm(j)
    R = zernike_radial(n, m, rho)
    norm = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
    if m > 0:
        return norm * R * np.cos(m * theta)
    elif m < 0:
        return norm * R * np.sin(abs(m) * theta)
    else:
        return norm * R

def load_valid_zernikes(csv_path, filter_str='525.02', only_qsun=False):
    """Load valid Z4..Z21 aberration vectors (in radians) from TuMag PD CSV."""
    df = pd.read_csv(csv_path, index_col=0)
    z_rows = [r for r in df.index if any(r.startswith(f"Z{i}") and (len(r)==len(f"Z{i}") or not r[len(f"Z{i}")].isdigit()) for i in range(4, 22))]

    valid_dict = {}
    for col in df.columns:
        if str(df.loc['Filter', col]).strip() != filter_str:
            continue
        if only_qsun and 'QSUN' not in col:
            continue
        vals = pd.to_numeric(df.loc[z_rows, col], errors='coerce').values
        if not np.isnan(vals).all():
            vals = np.nan_to_num(vals, nan=0.0)
            valid_dict[col] = vals
    return valid_dict

def generate_tumag_psf(
    psf_size=65,
    wavelength_nm=525.02,
    diameter_m=1.0,
    obscuration_ratio=0.324,
    pixel_scale_arcsec=0.0378,
    zernike_coeffs=None,
    grid_size=512
):
    """
    Generate physically scaled PSF for Sunrise III / TuMag via Fourier Optics.
    - If zernike_coeffs is None: returns diffraction-limited Airy PSF (with central obscuration).
    - If zernike_coeffs is provided (array of Z4..Z21 in radians): includes wavefront aberrations.
    """
    lam = wavelength_nm * 1e-9
    pix_rad = pixel_scale_arcsec * (np.pi / (180.0 * 3600.0))

    # Exact physical ratio between FFT grid size and pupil diameter in pixels
    padding_ratio = lam / (diameter_m * pix_rad)
    pupil_diameter_px = grid_size / padding_ratio
    pupil_radius_px = pupil_diameter_px / 2.0

    # Create pupil coordinate grid
    coords = np.arange(grid_size) - grid_size / 2.0
    X, Y = np.meshgrid(coords, coords)
    R_px = np.sqrt(X**2 + Y**2)
    rho = R_px / pupil_radius_px
    theta = np.arctan2(Y, X)

    # Annular aperture (Sunrise 1m primary + 0.324m secondary mirror)
    aperture = ((rho <= 1.0) & (rho >= obscuration_ratio)).astype(np.float64)

    # Wavefront phase error
    phase = np.zeros_like(rho)
    if zernike_coeffs is not None:
        mask = rho <= 1.0
        for idx, coeff in enumerate(zernike_coeffs):
            j = idx + 4  # Z4 to Z21
            if coeff != 0.0:
                phase[mask] += coeff * zernike_noll(j, rho[mask], theta[mask])

    pupil_func = aperture * np.exp(1j * phase)

    # Compute PSF via FFT
    otf_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_func)))
    psf_full = np.abs(otf_field)**2

    # Crop centered kernel of size psf_size x psf_size
    center = grid_size // 2
    half = psf_size // 2
    psf = psf_full[center - half : center + half + 1, center - half : center + half + 1]
    psf /= np.sum(psf)

    return psf.astype(np.float32)
