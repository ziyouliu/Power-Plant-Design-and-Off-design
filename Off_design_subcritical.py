import CoolProp.CoolProp as CP
import numpy as np
from Heat_exchanger_model import evaporator_off_design, condenser_off_design

working_fluid = 'R134a'
# Tc = CP.PropsSI('Tcrit', working_fluid)    # critical temperature, K
# pc = CP.PropsSI('Pcrit', working_fluid)    # critical pressure, Pa
geo_fluid = 'water'
cool_fluid = 'air'
T_max = 180 + 273.15   # safe operation temperature of R134a. 200 C is limit. Isopentane 400 C, control at 380 C.


class ORC(object):
    def __init__(self, T_prod, p_surf, m_gf, T_ambient, T_reinj, V_d, A_eva_d, m_wf_d, A_cond_d,
                 p_pump, deltaT_ap_cond, delta_T_subcool=2,
                 p_ambient=101325, p_cool=101325, fluid=working_fluid, geofluid=geo_fluid, cool_fluid=cool_fluid,
                 g=9.8, H=25,
                 v_wf_eva=2, v_wf_cond=15, v_gf=2, v_cool=20,
                 d_i=0.02, d_o=0.025, min_pinch_eva=5,
                 eta_t=0.8, eta_p=0.7, deltaP_eva=0, eta_t_mech=0.95, eta_p_mech=0.95,
                 convergence_threshold=0.01):
        self.fluid = fluid
        self.geofluid = geofluid
        self.cool_fluid = cool_fluid

        self.m_gf = m_gf                   # mass flow rate of geofluid, kg/s
        self.T_reinj = T_reinj + 273.15    # reinjection temperature, geofluid temperature after boiler, K.
        self.T_prod = T_prod + 273.15      # produced geofluid temperature, K
        self.p_surf = p_surf * 1e6         # wellhead pressure, pa. Input in MPa.

        self.T_ambient = T_ambient + 273.15      # ambient temperature, K. input in C.

        self.delta_T_subcool = delta_T_subcool   # subcooling.

        self.V_d = V_d                           # volumetric flow rate of working fluid at design point, m^3/s
        self.m_wf_d = m_wf_d                     # working fluid mass flow rate at design stage, kg/s
        self.A_eva_d = A_eva_d                   # surface area of the evaporator in design point, m^2.
        self.A_cond_d = A_cond_d                 # surface area of the condenser in design point, m^2.

        self.deltaT_ap_cond = deltaT_ap_cond     # approach point T difference of condenser, K. Determine condensing T. Optimize.
        self.p_pump = p_pump                     # pressure after pump, pa. Optimize.

        self.g = g                               # gravitational acceleration, 9.8 m/s^2
        self.H = H                               # pressure head of the condenser, 10 m. Li 2020.

        self.v_wf_eva = v_wf_eva                 # working fluid velocity in the evaporator, m/s.
        self.v_wf_cond = v_wf_cond               # working fluid velocity in the condenser, m/s.
        self.v_gf = v_gf                         # geofluid velocity in evaporator, m/s.
        self.v_cool = v_cool                     # cooling fluid velocity in condenser, m/s.
        self.d_i = d_i                           # inner diameter of tube, m.
        self.d_o = d_o                           # outer diameter of tube, m.
        self.min_pinch_eva = min_pinch_eva       # pinch point temperature difference of evaporator.

        self.deltaP_eva = deltaP_eva             # pressure drop in evaporator, 50kpa.
        self.p_ambient = p_ambient               # ambient pressure, pa.
        self.p_cool = p_cool                     # cooling pressure, Pa.
        self.eta_t = eta_t                       # turbine isentropic efficiency at design stage
        self.eta_p = eta_p                       # pump isentropic efficiency at design stage
        self.eta_t_mechanical = eta_t_mech       # mechanical efficiency of turbine/generator.
        self.eta_p_mechanical = eta_p_mech       # mechanical efficiency of pump.

        self.convergence_threshold = convergence_threshold   # Threshold for convergence of HX surface area

        # self.h_pw = CP.PropsSI('H', 'T', self.T_prod, 'P', self.p_surf.iloc[0], self.geofluid)    # enthalpy of production geofluid, J/kg.
        # self.s_pw = CP.PropsSI('S', 'T', self.T_prod, 'P', self.p_surf.iloc[0], self.geofluid)    # entropy of production geofluid, J/kg/K.
        # self.h0 = CP.PropsSI('H', 'T', self.T_ambient, 'P', self.p_ambient, self.geofluid)  # enthalpy of production geofluid, J/kg.
        # self.s0 = CP.PropsSI('S', 'T', self.T_ambient, 'P', self.p_ambient, self.geofluid)  # entralpy of production geofluid, J/kg/K.

        self.h_pw = CP.PropsSI('H', 'T', self.T_prod, 'P', self.p_surf, self.geofluid)    # enthalpy of production geofluid, J/kg.
        self.s_pw = CP.PropsSI('S', 'T', self.T_prod, 'P', self.p_surf, self.geofluid)    # entropy of production geofluid, J/kg/K.
        self.h0 = CP.PropsSI('H', 'T', self.T_ambient, 'P', self.p_ambient, self.geofluid)
        self.s0 = CP.PropsSI('S', 'T', self.T_ambient, 'P', self.p_ambient, self.geofluid)

        self.E_geo = self.m_gf * ((self.h_pw - self.h0) - self.T_ambient * (self.s_pw - self.s0))    # exergy of geofluid, W.

        self.h_wall = (self.d_o - self.d_i) / 2  # wall thickness, m.

        self.T_condensing = self.T_ambient + self.deltaT_ap_cond    # condensing temperature, K.
        self.p_condensing = CP.PropsSI('P', 'T', self.T_condensing, 'Q', 0, self.fluid)  # condensing pressure, pa.

    def pump(self):
        """
        isentropic pump (adiabatic and reversible)
        """
        # PUMP INLET
        self.T1 = self.T_condensing - self.delta_T_subcool    # condensing temperature at condenser outlet, K.
        self.p1 = self.p_condensing   # condensing pressure, pa.
        self.s1 = CP.PropsSI('S', 'T', self.T1, 'P', self.p1, self.fluid)  # entropy after condenser, or pump inlet, J/kg/K.
        self.h1 = CP.PropsSI('H', 'T', self.T1, 'P', self.p1, self.fluid)  # enthalpy after condenser, or pump inlet, J/kg.
        # PUMP OUTLET
        self.s2 = self.s1      # isentropic pump
        self.p2 = self.p_pump
        self.T2 = CP.PropsSI('T', 'P', self.p2, 'S', self.s2, self.fluid)  # temperature after pump, K.
        self.h2 = CP.PropsSI('H', 'P', self.p2, 'S', self.s2, self.fluid)  # enthalpy after pump, J/kg.

    def evaporator(self):
        """
        shell and tube heat exchanger.
        """
        # EVA INLET = PUMP OUTLET; PHE OUTLET = TURBINE INLET
        self.T_avg = (self.T_reinj + self.T_prod) / 2   # average temperature of geothermal injection and production well
        self.Cpw_avg = CP.PropsSI('C', 'T', self.T_avg, 'P', self.p_surf, self.geofluid)
        self.Q_geo = self.m_gf * self.Cpw_avg * (self.T_prod - self.T_reinj)    # geothermal heat input, W

    def turbine(self):
        # TURBINE INLET: assume pure steam after PHE to avoid corrosion
        self.p3 = self.p2 - self.deltaP_eva

        results_off_eva = evaporator_off_design(self.fluid, self.T2, self.p2, self.h2, self.v_wf_eva,
                                                self.geofluid, self.T_prod, self.T_reinj, self.Q_geo, self.p_surf,
                                                self.m_gf, self.v_gf, self.d_i, self.d_o, self.h_wall,
                                                self.min_pinch_eva, self.A_eva_d)

        self.T3 = results_off_eva[0]
        self.m_wf = results_off_eva[1]
        self.symb_eva = results_off_eva[2]            # relative error
        self.pinch_off_design = results_off_eva[3]    # pinch value in off-design stage

        self.s3 = CP.PropsSI('S', 'P', self.p3, 'T', self.T3, self.fluid)   # entropy before turbine, J/kg/K.
        self.h3 = CP.PropsSI('H', 'P', self.p3, 'T', self.T3, self.fluid)   # enthalpy before turbine, J/kg.
        self.rho3 = CP.PropsSI('D', 'P', self.p3, 'T', self.T3, self.fluid)  # working fluid density before turbine, kg/m3.
        self.V = self.m_wf / self.rho3  # volumetric flow rate of working fluid at design point, kg/m^3. The working fluid density use the density before turbine.
        # TURBINE OUTLET:
        self.p4 = self.p1
        self.s4 = self.s3     # isentropic turbine.
        self.h4 = CP.PropsSI('H', 'P', self.p4, 'S', self.s4, self.fluid)   # enthalpy after turbine, J/kg.
        self.T4 = CP.PropsSI('T', 'P', self.p4, 'S', self.s4, self.fluid)   # temperature after turbine, K.

    def condenser(self):
        self.Q_cool = self.m_wf * (self.h4 - self.h1)  # heat need to be taken by air, W.
        results_off_cond = condenser_off_design(self.fluid, self.p_condensing, self.h4, self.h1, self.m_wf, self.v_wf_cond,
                                                self.cool_fluid, self.Q_cool, self.T_ambient, self.T_condensing, self.p_cool,
                                                self.v_cool, self.d_i, self.d_o, self.h_wall, self.A_cond_d)

        self.T_out_cool = results_off_cond[0]
        self.m_cf = results_off_cond[1]
        self.symb_cond = results_off_cond[2]   # relative error

    def ORC_results_geothermal(self):
        """
        Only calculate condenser after evaporator converges to enhance efficiency
        """
        # Initialize
        self.pump()
        self.evaporator()
        self.turbine()

        # Check if evaporator converged based on the returned relative error
        evaporator_converged = abs(self.symb_eva) <= self.convergence_threshold

        # Only proceed with condenser calculation if evaporator converged
        if evaporator_converged:
            self.condenser()
            # Calculate efficiencies and outputs
            # Turbine efficiency at off-design condition
            self.eta_t_off = self.eta_t * np.sin(0.5 * np.pi * (self.V / self.V_d) ** 0.1)
            self.eta_p_off = 2 * self.eta_p * (self.V / self.V_d) - self.eta_p * (self.V / self.V_d) ** 2
            # WORK
            self.W_turbine = self.m_wf * (self.h3 - self.h4) * self.eta_t_off  # raw turbine output, W.
            self.W_pump = self.m_wf * (self.h2 - self.h1) / self.eta_p_off  # pump work, W

            if self.cool_fluid == 'seawater' or 'water':
                self.W_condenser = self.m_cf * self.g * self.H  # energy consumed by condenser, W, Li 2020.
            else:
                self.W_condenser = 150 * self.m_cf  # energy consumed by condenser, W, 0.15kW per kg/s of air

            # POWER
            self.W_output = self.W_turbine * self.eta_t_mechanical  # raw output, W
            self.W_parasitic = self.W_pump / self.eta_p_mechanical + self.W_condenser  # parasitic load, W
            self.W_elec = self.W_output - self.W_parasitic  # net electric output, W.
            self.W_net = self.W_elec / 1e6  # net electricity in MW.

            # EFFICIENCY
            self.Q_input = self.Q_geo
            self.E_input = self.E_geo
            self.eta_ORC = self.W_elec / self.Q_input  # 1st law efficiency
            self.exer_ORC = self.W_elec / self.E_input  # 2nd law efficiency

        else:
            # Only set these defaults if evaporator didn't converge
            self.symb_cond = 1.0
            self.T_out_cool = self.T_ambient
            self.m_cf = 0
            self.eta_t_off = 0
            # Set default values if calculations failed
            self.W_turbine = 0
            self.W_pump = 0
            self.W_condenser = 0
            self.W_output = 0
            self.W_parasitic = 0
            self.W_elec = 0
            self.W_net = 0
            self.eta_ORC = 0
            self.exer_ORC = 0

        return (
        self.m_wf, self.W_net, self.eta_ORC, self.exer_ORC, self.eta_t_off, self.T3, self.T_out_cool, self.m_cf,
        self.symb_eva, self.symb_cond, self.pinch_off_design)
