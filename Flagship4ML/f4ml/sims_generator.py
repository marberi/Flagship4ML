"""Module for generating simulated astronomical images."""

# Standard library imports
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

# Third party imports
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.modeling.functional_models import Moffat2D, Sersic2D
from scipy.signal import fftconvolve
from skimage.measure import block_reduce
from tqdm import tqdm

# Local imports
from Flagship4ML.f4ml.constants import DATA_DIR, PROJ_ROOT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CreateSimulatedImages:
    catalogue: str
    bands: list[str]
    crop_size: int = 60
    resolution: int = 10
    Ngals: int = 10
    add_poisson: bool = True
    add_psf: bool = True
    add_constant_background: bool = True
    num_exposures: int = 3
    use_dask: bool = False
    calibrate_flux: bool = True
    output_dir: Path = Path("/data/astro/scratch2/lcabayol/NFphotoz/data/CHFT_sims")

    def __post_init__(self):
        logger.info("Initializing CreateSimulatedImages")
        self.output_dir = Path(self.output_dir)
        if self.use_dask:
            import dask
            import dask.array as da
        # Load catalogue
        self.catalogue = pd.read_parquet(self.catalogue, engine="pyarrow")
        # Calculate PSF crop size
        self.crop_size_psf = self.crop_size / 2

        # Create grids for galaxy and PSF
        self.xgrid, self.ygrid = np.meshgrid(
            np.arange(0, self.resolution * self.crop_size, 1),
            np.arange(0, self.resolution * self.crop_size, 1),
        )
        self.psf_xgrid, self.psf_ygrid = np.meshgrid(
            np.arange(0, self.resolution * self.crop_size_psf, 1),
            np.arange(0, self.resolution * self.crop_size_psf, 1),
        )

    def _map_band_names(self, catalogue: pd.DataFrame) -> pd.DataFrame:
        """
        Map band names in the catalogue using the provided JSON mapping.

        Parameters
        ----------
        catalogue : pd.DataFrame
            DataFrame containing the astronomical catalogue data.

        Returns
        -------
        pd.DataFrame
            DataFrame with band names mapped according to the JSON mapping.
        """
        # Create a dictionary for mapping band names from the JSON mapping
        band_mapping = {
            key: value["band_name"] for key, value in self.json_band_photometry.items()
        }

        # Rename columns in the catalogue using the band_mapping dictionary
        catalogue = catalogue.rename(columns=band_mapping)

        return catalogue

    def _map_pau_bands(self, catalogue: pd.DataFrame) -> pd.DataFrame:
        """
        Map PAU narrow band names from _el suffix to standard format.

        Parameters
        ----------
        catalogue : pd.DataFrame
            DataFrame containing the astronomical catalogue data.

        Returns
        -------
        pd.DataFrame
            DataFrame with PAU band names mapped to standard format.
        """
        # Define PAU narrow bands
        pau_bands = [f"pau_nb{x}" for x in np.arange(455, 855, 10)]
        pau_bands_el = [f"pau_nb{x}_el" for x in np.arange(455, 855, 10)]
        pau_rename_map = dict(zip(pau_bands_el, pau_bands))

        # Rename columns in the catalogue
        catalogue = catalogue.rename(columns=pau_rename_map)

        return catalogue

    def _flux2mag(self, flux: np.ndarray) -> np.ndarray:
        """
        Convert flux to AB magnitude.

        Parameters
        ----------
        flux : np.ndarray
            Flux values to be converted.

        Returns
        -------
        np.ndarray
            Corresponding AB magnitudes.
        """
        mag = -2.5 * np.log10(flux) - 48.6
        return mag

    def _mag2e(self, mag: np.ndarray, zp: np.ndarray) -> np.ndarray:
        """Convert AB magnitudes to electrons using a zero-point value.

        Parameters
        ----------
        mag : np.ndarray
            AB magnitudes
        zp : np.ndarray
            Zero-point value for the conversion

        Returns
        -------
        np.ndarray
            Corresponding electron values
        """
        # temporary
        electrons = 10 ** (-0.4 * (mag - zp[None, :]))  # * self.exp_time[None,:]
        return electrons

    def _preprocess_catalogue(self, catalogue: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the astronomical catalogue.

        Steps:
        1. Filter galaxies based on observed redshift, bulge ellipticity, bulge r50, and disk r50.
        2. Sample a specified number of galaxies.
        3. Reset the index of the catalogue.

        Parameters:
        ----------
        catalogue : pd.DataFrame
            DataFrame containing the astronomical catalogue data.

        Returns:
        -------
        pd.DataFrame
            Preprocessed DataFrame.
        """
        filtered_catalogue = catalogue[
            (catalogue.observed_redshift_gal < 1)
            & (catalogue.bulge_ellipticity > 0)
            & (catalogue.bulge_r50 > 0)
            & (catalogue.disk_r50 > 0)
        ]

        sampled_catalogue = filtered_catalogue.sample(n=self.Ngals)

        return sampled_catalogue.reset_index(drop=True)

    def _get_photometry(self, catalogue: pd.DataFrame) -> pd.DataFrame:
        """Extract photometry from the catalogue and convert to fluxes.

        Parameters
        ----------
        catalogue : pd.DataFrame
            DataFrame containing the astronomical catalogue data.

        Returns
        -------
        pd.DataFrame
            Flux values for each photometric band.
        """
        photometry = catalogue[self.bands]

        # Convert magnitudes to fluxes for non-PAU bands
        if not any("pau" in band.lower() for band in self.bands):
            mags = self._flux2mag(np.abs(photometry))
            fluxes = self._mag2e(mags, self.zp)
        else:
            fluxes = photometry

        return fluxes

    def _get_morphology(self, catalogue: pd.DataFrame) -> pd.DataFrame:
        """
        Extract morphology parameters from the catalogue and return as a DataFrame.

        Args:
            catalogue (pd.DataFrame): DataFrame containing the astronomical catalogue data.

        Returns:
            pd.DataFrame: DataFrame of morphology parameters.
        """
        morphology_params = [
            "bulge_r50",
            "disk_r50",
            "bulge_nsersic",
            "disk_nsersic",
            "bulge_ellipticity",
            "disk_ellipticity",
            "bulge_fraction",
            "observed_redshift_gal",
            "disk_angle",
        ]
        morphology = catalogue[morphology_params].copy()

        # Handle disk angle conversion
        morphology["disk_angle"] = morphology["disk_angle"].abs()
        morphology["rotation_angle"] = morphology["disk_angle"] / 360 * 2 * np.pi
        morphology.drop("disk_angle", axis=1, inplace=True)

        # Rename columns for consistency
        column_mapping = {
            "bulge_r50": "hlr_b",
            "disk_r50": "hlr_d",
            "bulge_nsersic": "nsersic_bulge",
            "disk_nsersic": "nsersic_disk",
            "bulge_ellipticity": "ellip_bulge",
            "disk_ellipticity": "ellip_disk",
            "bulge_fraction": "bulge_disk_fraction",
            "observed_redshift_gal": "redshift",
        }
        morphology.rename(columns=column_mapping, inplace=True)

        return morphology

    def _get_zero_point_calibration(self) -> float:
        """Get a random zero-point calibration value."""
        return float(np.random.uniform(2, 4, 1))

    def _get_exp_time(self) -> np.ndarray:
        """
        Get exposure times for each band from the JSON mapping.

        Returns:
            np.ndarray: Array of exposure times for each band.
        """
        exp_times = {
            value["band_name"]: value["t_exp"]
            for value in self.json_band_photometry.values()
        }
        return np.array([exp_times[band] for band in self.bands])

    def _get_zp(self) -> np.ndarray:
        """
        Get zero-points for each band from the JSON mapping.

        Returns:
            np.ndarray: Array of zero-point values for each band.
        """
        zp = {
            value["band_name"]: value["ZP"]
            for value in self.json_band_photometry.values()
            if value["band_name"] in self.bands
        }
        return np.array([zp[band] for band in self.bands])

    def _get_pix_scale(self) -> dict[str, float]:
        """
        Get pixel scales for each band from the JSON mapping.

        Returns:
            dict[str, float]: Dictionary of pixel scales for each band.
        """
        pix_scales = {
            value["band_name"]: value["pix_scale"]
            for value in self.json_band_photometry.values()
            if value["band_name"] in self.bands
        }
        self.pix_scales = pix_scales
        return pix_scales

    def _get_psf_arcsec(self) -> dict[str, float]:
        """
        Get PSFs in arcseconds/pix for each band from the JSON mapping.

        Returns:
            dict[str, float]: Dictionary of PSF values for each band.
        """
        psfs = {
            value["band_name"]: value["psf"]
            for value in self.json_band_photometry.values()
            if value["band_name"] in self.bands
        }
        self.psfs = psfs
        return psfs

    def _simulate_exposure(
        self, photometry: pd.Series, morphology: pd.Series, band: str, zp: float = 1.0
    ) -> np.ndarray:
        """
        Simulate a galaxy image based on provided photometry and morphology parameters.

        Args:
            photometry: Photometry values for the galaxy in the specified band.
            morphology: Morphology parameters for the galaxy.
            band: Band for which the galaxy image is simulated.
            zp: Zero-point calibration value. Defaults to 1.0.

        Returns:
            Simulated galaxy image as a numpy array.
        """
        pix_scale = self.pix_scales[band]
        psf = self.psfs[band]

        sersic_bulge = Sersic2D(
            x_0=int(self.resolution * self.crop_size / 2),
            y_0=int(self.resolution * self.crop_size / 2),
            ellip=morphology.ellip_bulge,
            r_eff=self.resolution * morphology.hlr_b / pix_scale,
            n=morphology.nsersic_bulge,
            amplitude=1,
        )
        gal_bulge = sersic_bulge(self.xgrid, self.ygrid)

        sersic_disk = Sersic2D(
            x_0=int(self.resolution * self.crop_size / 2),
            y_0=int(self.resolution * self.crop_size / 2),
            ellip=morphology.ellip_disk,
            r_eff=self.resolution * morphology.hlr_d / pix_scale,
            n=morphology.nsersic_disk,
            amplitude=1,
        )
        gal_disk = sersic_disk(self.xgrid, self.ygrid)

        ib = self.bands.index(band)
        flux = photometry[band] * self.exp_time[ib] / zp
        flux_bulge = flux * morphology.bulge_disk_fraction
        flux_disk = flux * (1 - morphology.bulge_disk_fraction)

        gal_bulge = gal_bulge * flux_bulge / gal_bulge.sum() * self.resolution**2
        gal_disk = gal_disk * flux_disk / gal_disk.sum() * self.resolution**2

        gal = gal_bulge + gal_disk

        if self.add_constant_background:
            bkg = (
                np.random.uniform(1, 3)
                * self.exp_time[ib]
                / zp
                * np.ones(
                    (self.crop_size * self.resolution, self.crop_size * self.resolution)
                )
            )
            gal += bkg

        if self.add_psf:
            psf_size = self.resolution * psf / pix_scale
            gamma = psf_size / (2 * np.sqrt(2 ** (1 / 4.76) - 1))
            amplitude = (4.76 - 1) / (np.pi * gamma**2)
            moff = Moffat2D(
                amplitude=amplitude,
                x_0=int(self.resolution * self.crop_size_psf / 2),
                y_0=int(self.resolution * self.crop_size_psf / 2),
                gamma=gamma,
                alpha=4.76,
            )
            psf_grid = moff(self.psf_xgrid, self.psf_ygrid)
            gal = fftconvolve(gal, psf_grid, mode="same")

        if self.add_poisson:
            gal = np.random.poisson(gal)
        
        gal = block_reduce(gal, (self.resolution, self.resolution), np.mean)
        gal /= self.exp_time[ib]

        return gal

    def _create_simulated_galaxy(
        self, ii, band, exp, photometry_row, morphology_row, zp
    ):
        """
        Create and save simulated galaxy images along with metadata for a specific band and exposure.

        Parameters
        ----------
        ii : int
            Index of the galaxy in the catalogue.
        band : str
            Photometric band for which to simulate the galaxy image.
        exp : int
            Exposure number for this observation.
        photometry_row : pd.Series
            Series containing photometric measurements for the galaxy.
        morphology_row : pd.Series
            Series containing morphological parameters for the galaxy.
        zp : float
            Zero point calibration value to apply to the simulated flux.
        """
        output_file = self.output_dir / f"data_{ii}" / f"cutout_{band}_exp{exp}.npy"
        if not output_file.exists():
            # Simulate galaxy image
            gal = self._simulate_exposure(
                photometry_row, morphology_row, band=band, zp=zp
            )

            crop = gal[12:-12, 12:-12]#the length of the edge is 12, it is adjustable

            # Extract metadata
            metadata = np.c_[morphology_row["redshift"], photometry_row[band], zp]

            # Save simulated galaxy image and metadata
            np.save(output_file, crop)
            np.save(
                self.output_dir / f"data_{ii}" / f"metadata_{band}_exp{exp}.npy",
                metadata,
            )

    def _create_simulated_catalogue(self, Ngals: int) -> None:
        """
        Create a simulated catalogue of galaxy images without parallelization.

        Parameters
        ----------
        Ngals : int
            The number of galaxies to simulate in the catalogue.
        """
        photometry = self._get_photometry(self.catalogue)
        morphology = self._get_morphology(self.catalogue)

        for ii in tqdm(range(Ngals), desc="Creating simulated galaxies"):
            (self.output_dir / f"data_{ii}").mkdir(exist_ok=True)

            for band in self.bands:
                for e in range(self.num_exposures):
                    zp = (
                        self._get_zero_point_calibration() if self.calibrate_flux else 1
                    )

                    self._create_simulated_galaxy(
                        ii, band, e, photometry.iloc[ii], morphology.iloc[ii], zp
                    )
        logger.info(f"Created simulated catalogue with {Ngals} galaxies")

    def _create_simulated_catalogue_dask(self, n_gals: int) -> None:
        """
        Create and save a catalog of simulated galaxy images using Dask parallelization.

        Parameters
        ----------
        n_gals : int
            Number of galaxies to simulate.
        """
        photometry = self._get_photometry(self.catalogue)
        morphology = self._get_morphology(self.catalogue)

        delayed_tasks = []

        for ii in range(n_gals):
            (self.output_dir / f"data_{ii}").mkdir(exist_ok=True)

            for band in self.bands:
                for exp in range(self.num_exposures):
                    zp = (
                        self._get_zero_point_calibration() if self.calibrate_flux else 1
                    )
                    # Create delayed task with Series objects
                    task = dask.delayed(self._create_simulated_galaxy)(
                        ii,
                        band,
                        exp,
                        photometry.iloc[ii],  # Pass Series
                        morphology.iloc[ii],  # Pass Series
                        zp,
                    )
                    delayed_tasks.append(task)

        # Compute all tasks with a reasonable number of workers
        with dask.config.set(scheduler="processes", num_workers=4):
            dask.compute(*delayed_tasks)
        logger.info(f"Created simulated catalogue with {n_gals} galaxies using Dask")

    def generate_simulated_catalogue(self):
        # Load band photometry mapping from JSON file
        with open(DATA_DIR / "mapping.json", "r") as json_file:
            self.json_band_photometry = json.load(json_file)
        logger.info("Loaded band photometry mapping")

        # Preprocess and map band names in the catalogue
        self.catalogue = self._preprocess_catalogue(self.catalogue)
        self.catalogue = self._map_pau_bands(self.catalogue)
        self.catalogue = self._map_band_names(self.catalogue)
        logger.info("Preprocessed and mapped band names in catalogue")

        # Get various parameters
        self.pix_scale = self._get_pix_scale()
        self.psf = self._get_psf_arcsec()
        self.zp = self._get_zp()
        self.exp_time = self._get_exp_time()
        logger.info(
            "Retrieved pixel scale, PSF, zero point, and exposure time parameters"
        )

        # Create simulated catalogue using Dask or sequential method
        if self.use_dask:
            logger.info("Using Dask for parallel processing")
            self._create_simulated_catalogue_dask(self.Ngals)
        else:
            logger.info("Using sequential processing")
            self._create_simulated_catalogue(self.Ngals)
