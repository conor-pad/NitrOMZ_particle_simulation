# biopar.py
import dataclasses
from dataclasses import field

@dataclasses.dataclass
class BioPar:
    """
    Biogeochemical parameters based on nit_biopar_omz.m
    Includes dynamic stoichiometry from get_stoichiometry.m
    """
    # ── Organic Matter Composition (Anderson & Sarmiento 1994) ──
    k_hyd: float = 0.1 / 86400      # POC hydrolysis rate (1/d) -> Approx 10-day half-life for solid carbon
    stoch_a: float = 106.0  # C
    stoch_b: float = 175.0  # H
    stoch_c: float = 42.0   # O
    stoch_d: float = 16.0   # N
    
    # ── Derived Stoichiometric Coefficients ──
    # These are calculated automatically in __post_init__ below
    NCrem:  float = field(init=False)
    PCrem:  float = field(init=False)
    OCrem:  float = field(init=False)
    NCden1: float = field(init=False)
    NCden2: float = field(init=False)
    NCden3: float = field(init=False)

    def __post_init__(self):
        """Calculates stoichiometry of redox reactions based on C:H:O:N:P"""
        a, b, c, d = self.stoch_a, self.stoch_b, self.stoch_c, self.stoch_d
        
        # number of electrons required to oxidize organic matter
        Corg_e = 4*a + b - 2*c - 3*d + 5
        
        O2toH2O_e = 4
        HNO3toHNO2_e = 2
        HNO2toN2O_e = 2
        N2OtoN2_e = 2
        
        self.OCrem  = (Corg_e / O2toH2O_e) / a       # molO2 / molC
        self.NCrem  = d / a                          # molNH4 / molC
        self.PCrem  = 1.0 / a                        # molP / molC
        self.NCden1 = (Corg_e / HNO3toHNO2_e) / a    # Nitrate to Carbon ratio
        self.NCden2 = (Corg_e / HNO2toN2O_e) / a     # Nitrite to Carbon ratio
        self.NCden3 = (Corg_e / N2OtoN2_e) / a       # N2O to Carbon ratio

   # ── Ammonification ──
    krem:   float = 0.08 / 86400          
    KO2Rem: float = 0.5           

    # ── Ammonium oxidation (Ammox: NH4 -> NO2) ──
    kAo:    float = 0.04556 / 86400       
    KNH4Ao: float = 0.0272        
    KO2Ao:  float = 0.333         

    # ── Nitrite oxidation (Nitrox: NO2 -> NO3) ──
    kNo:    float = 0.255 / 86400         
    KNO2No: float = 0.0272        
    KO2No:  float = 0.778         

    # ── Denitrification ──
    kDen1:    float = (0.08 / 2.0) / 86400  
    KO2Den1:  float = 1.0         
    KNO3Den1: float = 0.5         

    kDen2:    float = (0.08 / 6.0) / 86400  
    KO2Den2:  float = 0.3         
    KNO2Den2: float = 0.5         

    kDen3:    float = (0.08 / 3.0) / 86400  
    KO2Den3:  float = 0.0292      
    KN2ODen3: float = 0.02        

    # ── Anammox ──
    kAx:    float = 0.02 / 86400          
    KNH4Ax: float = 0.0274        
    KNO2Ax: float = 0.5           
    KO2Ax:  float = 0.886         
