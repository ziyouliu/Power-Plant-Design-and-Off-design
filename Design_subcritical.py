import numpy as np
from iapws import SeaWater
import CoolProp.CoolProp as CP
from Heat_exchanger_model import calculate_evaporator_area, calculate_condenser_area

working_fluid = 'R134a'
# Tc = CP.PropsSI('Tcrit', working_fluid)    # critical temperature, K
# pc = CP.PropsSI('Pcrit', working_fluid)    # critical pressure, Pa
geo_fluid = 'water'
cool_fluid = 'air'
T_max = 180 + 273.15   # safe operation temperature. For R134a, 200 C is limit. Isopentane 400 C, control at 380 C.


class ORC(object):
    """
    assumptions and boundary conditions:
    - friction and heat losses are neglected.
    - evaporator and condenser efficiency is 100 %.
    - 1 MW = 1000 kW = 1e6 J/s
    - No recuperator
    - No pressure drop in the heat exchangers
    - Subcooling can be considered
    """

    def __init__(self, T_prod, p_surf, m_gf, T_ambient, T_reinj,
                 p_pump, deltaT_ap_cond, deltaT_pp_cond, deltaT_sh, delta_T_subcool=2,
                 p_ambient=101325, p_cool=101325, fluid=working_fluid, geofluid=geo_fluid, cool_fluid=cool_fluid,
                 g=9.8, H=25,
                 v_wf_eva=2, v_wf_cond=15, v_gf=2, v_cool=20, d_i=0.02, d_o=0.025, min_pinch_eva=5,
                 eta_t=0.8, eta_p=0.7, deltaP_eva=0, eta_t_mech=0.95, eta_p_mech=0.95):

        self.fluid = fluid
        self.geofluid = geofluid
        self.cool_fluid = cool_fluid
        # Given and decision parameters of geothermal reservoir
        self.m_gf = m_gf                         # mass flow rate of geofluid, kg/s
        self.T_reinj = T_reinj + 273.15          # reinjection temperature, geofluid temperature after boiler, K.
        self.T_prod = T_prod + 273.15            # produced geofluid temperature, K
        self.p_surf = p_surf * 1e6               # wellhead pressure, pa. Input in MPa.

        self.T_ambient = T_ambient + 273.15      # ambient temperature, K. input in C.

        self.deltaT_ap_cond = deltaT_ap_cond     # approach point T difference of condenser, K. Determine condensing T. Optimized in design stage.
        self.deltaT_pp_cond = deltaT_pp_cond     # pinch point T difference of condenser, K. Determine outlet T of condenser. Optimized in design stage.
        self.p_pump = p_pump                     # pressure after pump, pa. Optimized in design stage.
        self.deltaT_sh = deltaT_sh               # superheat degree. Optimized in design stage.

        self.delta_T_subcool = delta_T_subcool   # subcooling degree.

        self.g = g                               # gravitational acceleration, 9.8 m/s^2
        self.H = H                               # pressure head of the condenser for liquid cooling fluid. Li 2020.

        self.v_wf_eva = v_wf_eva                 # working fluid velocity in the evaporator, m/s.
        self.v_wf_cond = v_wf_cond               # working fluid velocity in the condenser, m/s.
        self.v_gf = v_gf                         # geofluid velocity in evaporator, m/s.
        self.v_cool = v_cool                     # cooling fluid velocity in condenser, m/s.
        self.d_i = d_i                           # inner diameter of tube, m.
        self.d_o = d_o                           # outer diameter of tube, m.
        self.min_pinch_eva = min_pinch_eva       # minimum pinch point temperature difference of evaporator.

        self.deltaP_eva = deltaP_eva             # pressure drop in evaporator.
        self.p_ambient = p_ambient               # ambient pressure, pa.
        self.p_cool = p_cool                     # cooling pressure, Pa.
        self.eta_t = eta_t                       # turbine isentropic efficiency at design stage
        self.eta_p = eta_p                       # pump isentropic efficiency at design stage
        self.eta_t_mechanical = eta_t_mech       # mechanical efficiency of turbine/generator.
        self.eta_p_mechanical = eta_p_mech       # mechanical efficiency of pump.

        # self.h_pw = CP.PropsSI('H', 'T', self.T_prod, 'P', self.p_surf.iloc[0], self.geofluid)    # enthalpy of production geofluid, J/kg.
        # self.s_pw = CP.PropsSI('S', 'T', self.T_prod, 'P', self.p_surf.iloc[0], self.geofluid)    # entropy of production geofluid, J/kg/K.
        # self.h0 = CP.PropsSI('H', 'T', self.T_ambient, 'P', self.p_ambient, self.geofluid)  # enthalpy of production geofluid, J/kg.
        # self.s0 = CP.PropsSI('S', 'T', self.T_ambient, 'P', self.p_ambient, self.geofluid)  # entralpy of production geofluid, J/kg/K.

        self.h_wall = (self.d_o - self.d_i) / 2  # wall thickness, m.

        self.h_pw = CP.PropsSI('H', 'T', self.T_prod, 'P', self.p_surf, self.geofluid)    # enthalpy of production geofluid, J/kg.
        self.s_pw = CP.PropsSI('S', 'T', self.T_prod, 'P', self.p_surf, self.geofluid)    # entropy of production geofluid, J/kg/K.
        self.h0 = CP.PropsSI('H', 'T', self.T_ambient, 'P', self.p_ambient, self.geofluid)
        self.s0 = CP.PropsSI('S', 'T', self.T_ambient, 'P', self.p_ambient, self.geofluid)

        self.E_geo = self.m_gf * ((self.h_pw - self.h0) - self.T_ambient * (self.s_pw - self.s0))    # exergy of geofluid, W.

        self.T_condensing = self.T_ambient + self.deltaT_ap_cond    # condensing temperature, K.
        self.p_condensing = CP.PropsSI('P', 'T', self.T_condensing, 'Q', 0, self.fluid)  # condensing pressure, pa.
        self.T_out_cond = self.T_condensing - self.deltaT_pp_cond     # cooling fluid temperature at condenser outlet.

    def pump(self):
        """
        isentropic pump (adiabatic and reversible)
        """
        # PUMP INLET
        self.T1 = self.T_condensing - self.delta_T_subcool   # condensing temperature at condenser outlet, with subcooling, K.
        self.p1 = self.p_condensing   # condensing pressure, pa.
        self.s1 = CP.PropsSI('S', 'T', self.T1, 'P', self.p1, self.fluid)  # entropy after condenser, or pump inlet, J/kg/K.
        self.h1 = CP.PropsSI('H', 'T', self.T1, 'P', self.p1, self.fluid)  # enthalpy after condenser, or pump inlet, J/kg.
        # PUMP OUTLET
        self.s2 = self.s1      # isentropic pump
        self.p2 = self.p_pump
        self.T2 = CP.PropsSI('T', 'P', self.p2, 'S', self.s2, self.fluid)  # temperature after pump, K.
        self.h2 = CP.PropsSI('H', 'P', self.p2, 'S', self.s2, self.fluid)  # enthalpy after pump, J/kg.

    def turbine(self):
        # TURBINE INLET: superheated state to assure no liquid droplet. The minimum superheat degree can be 0.01.
        # therefore, literally speaking, there is no saturated ORC anymore. If the optimal superheated degree is
        # close to 0, then we can consider it as saturated ORC.
        self.p3 = self.p2 - self.deltaP_eva
        self.T_sat = CP.PropsSI('T', 'P', self.p3, 'Q', 1, self.fluid)   # saturate temperature, K.
        self.T3 = self.T_sat + self.deltaT_sh
        '''
        add a constraint here. T3 < T_prod - 10, 10 can be changed. The temperatures shall not be too close.
        '''
        self.s3 = CP.PropsSI('S', 'P', self.p3, 'T', self.T3, self.fluid)   # entropy before turbine, J/kg/K.
        self.h3 = CP.PropsSI('H', 'P', self.p3, 'T', self.T3, self.fluid)   # enthalpy before turbine, J/kg.
        self.rho3 = CP.PropsSI('D', 'P', self.p3, 'T', self.T3, self.fluid)  # working fluid density before turbine, kg/m3.
        # TURBINE OUTLET:
        self.p4 = self.p1
        self.s4 = self.s3     # isentropic turbine.
        self.h4 = CP.PropsSI('H', 'P', self.p4, 'S', self.s4, self.fluid)   # enthalpy after turbine, J/kg.
        self.T4 = CP.PropsSI('T', 'P', self.p4, 'S', self.s4, self.fluid)   # temperature after turbine, K.

    def evaporator(self):
        """
        shell and tube heat exchanger.
        """
        # EVAPORATOR INLET = PUMP OUTLET; EVAPORATOR OUTLET = TURBINE INLET
        self.T_avg = (self.T_reinj + self.T_prod) / 2   # average temperature of geothermal injection and production well
        self.Cpw_avg = CP.PropsSI('C', 'T', self.T_avg, 'P', self.p_surf, self.geofluid)
        self.Q_geo = self.m_gf * self.Cpw_avg * (self.T_prod - self.T_reinj)    # geothermal heat input, W

        self.m_wf = self.Q_geo / (self.h3 - self.h2)    # working fluid mass rate, kg/s, maintaining constant reinjection temperature.
        self.V_d = self.m_wf / self.rho3     # volumetric flow rate of working fluid at design point, m^3/s. The working fluid density use the density before turbine.

        # Calculate evaporator surface area, m^2
        self.A_eva, self.pinch_value = calculate_evaporator_area(self.fluid, self.T2, self.T3, self.h2, self.h3, self.p2,
                                                                 self.m_wf, self.v_wf_eva, self.geofluid, self.T_prod,
                                                                 self.T_reinj, self.p_surf, self.m_gf, self.v_gf,
                                                                 self.d_i, self.d_o, self.h_wall, self.min_pinch_eva)

    def condenser(self):
        """
        use air cooling condenser (ACC) or seawater
        inlet temperature is ambient temperature
        """
        self.Q_cool = self.m_wf * (self.h4 - self.h1)  # heat need to be taken by air, W.
        T_avg = (self.T_ambient + self.T_out_cond) / 2

        if self.cool_fluid == 'seawater':
            sw = SeaWater(T=T_avg, P=0.1, S=0.035)
            self.Cp_c = sw.cp * 1000
        else:
            self.Cp_c = CP.PropsSI('C', 'T', T_avg, 'P', self.p_ambient, self.cool_fluid)

        self.m_cf = self.Q_cool / (self.Cp_c * (self.T_out_cond - self.T_ambient))   # air mass rate, kg/s.
        self.A_cond = calculate_condenser_area(self.fluid, self.h4, self.h1, self.p_condensing, self.m_wf,
                                               self.v_wf_cond, self.cool_fluid, self.T_ambient, self.T_out_cond,
                                               self.p_cool, self.v_cool, self.d_i, self.d_o, self.h_wall)

    def cal_fp(self, p):
        p = p / 1e6
        if p < 0.6:
            Fp = 1
        else:
            C1, C2, C3 = -0.00164, -0.00627, 0.0123
            Fp = 10 ** (C1 + C2 * np.log10(10 * p - 1) + C3 * np.log10(10 * p - 1) * np.log10(10 * p - 1))

        return Fp

    def cost(self, W_t, W_p, A_eva, A_cond, W_elec):
        """
        calculate the equipment costs of the power plant
        """
        K1_HX, K2_HX, K3_HX = 4.3247, -0.303, 0.1634    # HX, including evaporator and condenser.
        K1_T, K2_T, K3_T = 2.7051, 1.4398, -0.1776   # Turbine
        K1_P, K2_P, K3_P = 3.3892, 0.0536, 0.1538    # Pump
        B1, B2 = 1.63, 1.66
        FM = 1.8

        Cp0_eva = 10 ** (K1_HX + K2_HX * np.log10(A_eva) + K3_HX * np.log10(A_eva) * np.log10(A_eva))
        Cp0_cond = 10 ** (K1_HX + K2_HX * np.log10(A_cond) + K3_HX * np.log10(A_cond) * np.log10(A_cond))
        Cp0_T = 10 ** (K1_T + K2_T * np.log10(W_t / 1000) + K3_T * np.log10(W_t / 1000) * np.log10(W_t / 1000))
        Cp0_P = 10 ** (K1_P + K2_P * np.log10(W_p / 1000) + K3_P * np.log10(W_p / 1000) * np.log10(W_p / 1000))

        FBM_eva = B1 + B2 * FM * self.cal_fp(self.p_pump)
        FBM_cond = B1 + B2 * FM * self.cal_fp(self.p_condensing)
        FBM_T = B1 + B2 * FM * self.cal_fp(self.p_pump)
        FBM_P = B1 + B2 * FM * self.cal_fp(self.p_pump)

        PEC_eva = Cp0_eva * FBM_eva
        PEC_cond = Cp0_cond * FBM_cond
        PEC_T = Cp0_T * FBM_T
        PEC_P = Cp0_P * FBM_P

        PEC_sys = 6.32 * (PEC_eva + PEC_cond + PEC_T + PEC_P) * 1.5

        SIC = (PEC_sys / (W_elec / 1000)) / 1000   # k$/kW
        # SIC = PEC_sys / 1000  # k$

        return SIC

    def ORC_results_geothermal(self):
        self.pump()
        self.turbine()
        self.evaporator()
        self.condenser()

        if self.A_eva <= 0 or self.A_cond <= 0:
            # Only set these defaults if evaporator or condenser either doesn't converge
            self.symb_cond = 1.0
            self.T_out_cool = self.T_ambient
            self.m_cf = 0.0
            self.eta_t_off = 0
            self.W_turbine = 0
            self.W_pump = 0
            self.W_condenser = 0
            self.W_output = 0
            self.W_parasitic = 0
            self.W_elec = 0
            self.W_net = 0
            self.eta_ORC = 0
            self.exer_ORC = 0
            self.sic = 1e8
        else:
            # WORK
            self.W_turbine = self.m_wf * (self.h3 - self.h4) * self.eta_t    # raw turbine output, W.
            self.W_pump = self.m_wf * (self.h2 - self.h1) / self.eta_p       # pump work, W

            if self.cool_fluid == 'seawater' or 'water':
                self.W_condenser = self.m_cf * self.g * self.H  # energy consumed by condenser, W, Li 2020.
            else:
                self.W_condenser = 150 * self.m_cf    # energy consumed by condenser, W, 0.15kW per kg/s of air

            # POWER
            self.W_output = self.W_turbine * self.eta_t_mechanical   # raw output, W
            self.W_parasitic = self.W_pump / self.eta_p_mechanical + self.W_condenser   # parasitic load, W
            self.W_elec = self.W_output - self.W_parasitic   # net electric input, W.
            self.W_net = self.W_elec / 1e6       # net electricity in MW.
            # EFFICIENCY
            self.Q_input = self.Q_geo
            self.E_input = self.E_geo
            self.eta_ORC = self.W_elec / self.Q_input    # 1st law efficiency
            self.exer_ORC = self.W_elec / self.E_input   # 2nd law efficiency

            if self.W_elec <= 0 or self.W_output <= 0 or self.W_parasitic <= 0:
                self.sic = 1e8
            else:
                self.sic = self.cost(self.W_turbine, self.W_pump, self.A_eva, self.A_cond, self.W_elec)

        return (self.m_wf, self.W_net, self.eta_ORC, self.exer_ORC, self.V_d, self.A_eva,
                self.pinch_value, self.A_cond, self.m_cf, self.sic)
