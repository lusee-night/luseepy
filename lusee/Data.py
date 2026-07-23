import json

import  numpy           as np
import  fitsio
import astropy.units as u
from astropy.time import Time, TimeDelta
from lunarsky import MoonLocation
from    .Observation import Observation
from .LunarCalendar  import LunarCalendar
from .Throughput import Throughput
from .LabeledArray import label, FRAME_TOPO

class ThroughputBeam:
    """
    Dummy class that contains complex impedance for Throughput class

    :param freq: Frequency
    :type freq: float
    :param Z: Complex impedance
    :type Z: numpy array[complex]
    """
    ## dummy class to carry impedance for throughput
    def __init__(self, freq,Z):
        self.freq = freq
        self.Z = Z

class Data(Observation):
    """
    Class that holds data from an observation

    :param filename: Filename of observation to load
    :type filename: str
    :param throughput: Front-end throughput parameters
    :type throughput: class
    """
    
    def __init__(self, filename, throughput=None):
        header = dict(fitsio.read_header(filename))
        version = header['VERSION']
        if float(version) >= 3:
            self._init_covariance_fits(filename, header)
            return
        lunar_day    = header['LUNAR_DAY']
        lun_lat_deg  = header['LUN_LAT_DEG']
        lun_long_deg = header['LUN_LONG_DEG']
        lun_height_m = header['LUN_HEIGHT_M']
        deltaT_sec   = header['DELTAT_SEC']
        super().__init__(lunar_day, lun_lat_deg, lun_long_deg, lun_height_m, deltaT_sec)
        fits = fitsio.FITS(filename,'r')
        self.data = fits['data'].read()
        # Units of the raw 'data' array.  Simulator FITS output is brightness
        # temperature in Kelvin; an HDF5 ingest path may set this to e.g.
        # "counts" before gain conversion.
        self.data_units = "K"
        self.freq = fits['freq'].read()
        combinations = fits['combinations'].read()
        cc = 0
        comb2ndx = {}
        NBeams = 0
        for i,j in combinations:
            comb2ndx [(i,j)] = cc
            Nbeams = max(NBeams,i,j)
            if i==j:
                cc+=1
            else:
                cc+=2
        self.comb2ndx = comb2ndx
        # sanity checks
        if (i==j):
            assert(cc == self.data.shape[1])
        else:
            assert(cc == self.data.shape[1]+1)
            
        assert (len(self.times) == self.data.shape[0])
        assert (len(self.freq) == self.data.shape[2])
        
        Nbeams +=1
        self.ZRe = []
        self.ZIm = []
        for i in range(Nbeams):
            self.ZRe.append (fits[f'ZRe_{i}'].read())
            self.ZIm.append (fits[f'ZIm_{i}'].read())
        self.Nbeams = NBeams
        self.Nfreq = len(self.freq)
        self.Ntimes = len(self.times)
        self.NComb = len(self.comb2ndx)
        tbeam = ThroughputBeam(self.freq, self.ZRe[0]+1j*self.ZIm[0])
        ## this might need fixing.
        self.T = [Throughput(beam=tbeam) if throughput is None else throughput]*Nbeams
        self.T2Vsq = [T.T2Vsq(self.freq) for T in self.T]

    def _init_covariance_fits(self, filename, header):
        """Initialize the explicit-time, physical-covariance FITS format."""
        provenance_keys = (
            "ENGINE",
            "RESPONSE",
            "RESPHASH",
            "RESPVAL",
            "FREQINT",
            "RECMODEL",
            "RECCHANS",
            "RECSRC",
            "SKYMODEL",
            "SKYFRAME",
            "SKYSRC",
            "LUSEEVER",
            "CROVER",
            "S2FFTVER",
            "CLOCKSRC",
            "SCALEASM",
        )
        self.provenance = {
            key: header[key]
            for key in provenance_keys
            if key in header
        }
        self.response_provenance = {
            "path": header.get("RESPONSE"),
            "content_hash": header.get("RESPHASH"),
            "validated": bool(header.get("RESPVAL", False)),
            "frequency_interpolation": header.get("FREQINT"),
        }
        self.receiver_provenance = {
            "model": header.get("RECMODEL"),
            "channels": header.get("RECCHANS"),
            "source": header.get("RECSRC"),
        }
        self.sky_provenance = {
            "model": header.get("SKYMODEL"),
            "frame": header.get("SKYFRAME"),
            "source": header.get("SKYSRC"),
        }
        self.software_versions = {
            "lusee": header.get("LUSEEVER"),
            "croissant": header.get("CROVER"),
            "s2fft": header.get("S2FFTVER"),
        }
        fits = fitsio.FITS(filename, "r")
        data_header = dict(fits["data"].read_header())
        data_units = str(data_header.get("BUNIT", "")).strip()
        if data_units != "V^2/Hz":
            raise ValueError(
                f"Covariance data BUNIT must be 'V^2/Hz'; got "
                f"{data_units!r}."
            )
        freq_header = dict(fits["freq"].read_header())
        if str(freq_header.get("BUNIT", "")).strip() != "MHz":
            raise ValueError("Covariance frequency HDU must have BUNIT='MHz'.")
        time_header = dict(fits["time"].read_header())
        if str(time_header.get("BUNIT", "")).strip() != "d":
            raise ValueError("Covariance time HDU must have BUNIT='d'.")
        time_scale = str(
            time_header.get("TIMESYS", header.get("TIMESYS", ""))
        ).strip().lower()
        if not time_scale:
            raise ValueError("Covariance time HDU is missing TIMESYS.")
        time_values = np.asarray(fits["time"].read(), dtype=np.float64)
        exact_times = Time(time_values, format="mjd", scale=time_scale)
        if exact_times.isscalar:
            exact_times = Time([exact_times])

        delta_t = float(header.get("DELTAT_SEC", 0.0))
        if delta_t <= 0 and len(exact_times) > 1:
            delta_t = float(np.median(np.diff(exact_times.tdb.jd)) * 86400)
        if delta_t <= 0:
            delta_t = 1.0
        self.lun_lat_deg = float(header["LUN_LAT_DEG"])
        self.lun_long_deg = float(header["LUN_LONG_DEG"])
        self.lun_height_m = float(header["LUN_HEIGHT_M"])
        self.lun_lat = np.radians(self.lun_lat_deg)
        self.lun_long = np.radians(self.lun_long_deg)
        self.loc = MoonLocation.from_selenodetic(
            lon=self.lun_long_deg,
            lat=self.lun_lat_deg,
            height=self.lun_height_m,
        )
        self.deltaT_sec = delta_t
        self.deltaT = TimeDelta(delta_t * u.s)
        self.calibrator_tracks = []
        self.times = exact_times
        self.time_start = exact_times[0]
        self.time_end = exact_times[-1]
        self.time_range = header.get("LUNAR_DAY", "stored-time-axis")
        self.data = fits["data"].read()
        self.data_units = data_units
        self.freq = np.asarray(fits["freq"].read(), dtype=np.float64)
        product_data = fits["products"].read()
        if product_data.dtype.fields and "label" in product_data.dtype.fields:
            raw_labels = product_data["label"]
        else:
            raw_labels = product_data
        self.product_labels = tuple(
            value.decode("ascii").strip()
            if isinstance(value, (bytes, np.bytes_))
            else str(value).strip()
            for value in raw_labels
        )
        self.product_index = {
            label: index for index, label in enumerate(self.product_labels)
        }
        self.comb2ndx = {}
        for a in range(4):
            for b in range(a, 4):
                label = f"{a}{b}R"
                if label in self.product_index:
                    self.comb2ndx[(a, b)] = self.product_index[label]

        def read_complex(name, units):
            real_header = dict(fits[f"{name}_real"].read_header())
            imag_header = dict(fits[f"{name}_imag"].read_header())
            real_unit = str(real_header.get("BUNIT", "")).strip()
            imag_unit = str(imag_header.get("BUNIT", "")).strip()
            if real_unit != units or imag_unit != units:
                raise ValueError(
                    f"{name} real/imag HDUs must both have BUNIT={units!r}."
                )
            real = fits[f"{name}_real"].read()
            imag = fits[f"{name}_imag"].read()
            if real.shape != imag.shape:
                raise ValueError(
                    f"{name} real/imag HDUs have different shapes."
                )
            expected = (len(self.freq), 4, 4)
            if real.shape != expected:
                raise ValueError(
                    f"{name} must have target-aligned shape {expected}; "
                    f"got {real.shape}."
                )
            return real + 1j * imag

        self.ZA = read_complex("ZA", "Ohm")
        self.ZL = read_complex("ZL", "Ohm")
        self.M = read_complex("M", "1")
        self.Rsky = read_complex("Rsky", "Ohm")
        self.Rmoon = read_complex("Rmoon", "Ohm")
        self.blackbody_normalization = read_complex(
            "blackbody_normalization",
            "V^2/(Hz K)",
        )
        receiver_payload = np.asarray(
            fits["receiver_params"].read(),
            dtype=np.uint8,
        ).tobytes()
        self.receiver_params = json.loads(
            receiver_payload.decode("utf-8")
        )
        self.receiver_provenance["params"] = self.receiver_params
        fits.close()
        if self.data.shape != (
            len(self.times),
            len(self.product_labels),
            len(self.freq),
        ):
            raise ValueError(
                "Covariance data shape is inconsistent with time, product, "
                "or frequency axes."
            )
        self._format_version = 3
        self.Nbeams = 4
        self.Nfreq = len(self.freq)
        self.Ntimes = len(self.times)
        self.NComb = len(self.comb2ndx)

    def __getitem__(self, req):
        # Can do things like
        #   O[0,"01R",:]
        # or 
        #   O[0,(0,1,'R'),:]
        
        if getattr(self, "_format_version", None) == 3:
            return self._getitem_covariance(req)
        day,comb,freq = req
        fact = np.ones(self.Nfreq) # frequency scaling
        if type(comb) == str:
            if comb[0]=="-":
                fact=-1
                comb=comb[1:]
            i = int(comb[0])
            j = int(comb[1])
            if len(comb)>=3:
                what = comb[2:]
            else:
                what = 'R' if (i==j) else 'C'
        else:
            i,j = comb
            if len(comb)==3:
                what = comb[2]
            else:
                what = 'R' if (i==j) else 'C'

       
        vwhat = what[1:]
        what = what[0] 
    
        ndx = self.comb2ndx[(i,j)]
        if what == "R":
            toret = self.data[day,ndx,freq]
        if what == "I":
            assert (i!=j)
            toret= self.data[day,ndx+1,freq]
        if what == "C":
            assert (i!=j)
            toret=self.data[day,ndx,freq]+1j*self.data[day,ndx+1,freq]


        if vwhat == "":
            return label(toret, units=self.data_units, frame=FRAME_TOPO)
        elif vwhat == "V":
                ## ffact can be scalar +1 or -1
                # T2Vsq = 4 kB Re(Z) Gamma^2 is V^2/(Hz K), so multiplying a
                # Kelvin (auto/cross) value by sqrt(T2Vsq_i T2Vsq_j) yields a
                # voltage power spectral density in V^2/Hz, not amplitude.
                T2V = np.sqrt(self.T2Vsq[i]*self.T2Vsq[j])[freq]
                if toret.ndim == 1:
                    return label(toret*T2V, units="V^2/Hz", frame=FRAME_TOPO)
                else:
                    return label(toret*T2V[None,:], units="V^2/Hz", frame=FRAME_TOPO)
        else:
            raise NotImplementedError

        # Should not get here.
        raise NotImplementedError

    def _getitem_covariance(self, req):
        """Index physical covariance channels with optional K-equivalent view."""
        day, comb, freq = req
        sign = 1.0
        if isinstance(comb, str):
            token = comb.upper()
            if token.startswith("-"):
                sign = -1.0
                token = token[1:]
            if len(token) < 2:
                raise ValueError(f"Invalid product request {comb!r}.")
            a, b = int(token[0]), int(token[1])
            remainder = token[2:]
        else:
            values = tuple(comb)
            if len(values) not in {2, 3}:
                raise ValueError("Tuple product requests have length 2 or 3.")
            a, b = int(values[0]), int(values[1])
            remainder = "" if len(values) == 2 else str(values[2]).upper()
        if not (0 <= a < 4 and 0 <= b < 4):
            raise ValueError("Product port indices must lie in [0, 3].")

        view = ""
        if remainder.endswith("K"):
            view = "K"
            remainder = remainder[:-1]
        elif remainder.endswith("V"):
            # Legacy V suffix is a documented no-op for already physical PSD.
            view = "V"
            remainder = remainder[:-1]
        component = remainder or ("R" if a == b else "C")
        if component not in {"R", "I", "C"}:
            raise ValueError(f"Invalid covariance component {component!r}.")
        if a == b and component == "I":
            raise ValueError("Auto products have no imaginary channel.")

        lo, hi = min(a, b), max(a, b)
        real_label = f"{lo}{hi}R"
        if real_label not in self.product_index:
            raise KeyError(f"Product {real_label!r} is not stored.")
        real = self.data[day, self.product_index[real_label], freq]
        if lo == hi:
            complex_value = real
        else:
            imag_label = f"{lo}{hi}I"
            if imag_label not in self.product_index:
                raise KeyError(f"Product {imag_label!r} is not stored.")
            imag = self.data[day, self.product_index[imag_label], freq]
            complex_value = real + 1j * imag
            if a > b:
                complex_value = np.conjugate(complex_value)
        if view == "K":
            normalization = self.blackbody_normalization[freq, a, b]
            complex_value = complex_value / normalization
        if component == "R":
            result = np.real(complex_value)
        elif component == "I":
            result = np.imag(complex_value)
        else:
            result = complex_value
        units = "K" if view == "K" else "V^2/Hz"
        return label(sign * result, units=units, frame=FRAME_TOPO)


            
            
    
