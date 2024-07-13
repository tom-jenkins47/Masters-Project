import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.optimize
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy import stats
import statsmodels.api as sm
from statistics import linear_regression
import os
import re
import pandas as pd
plt.style.use('ggplot')

"""
To do:
    - make all writing to outfiles use Pandas
    - write plot functions to make things less confusing
    - combine pressure and Up outfiles 
    - plot pressure vs. particle velocity
    - try and modify pulse using file Martin sent
    - add copper data to other energy plots
"""

class LaserPulse:
    
    def __init__(self, pulse_profile, pulse_energy, spot_size, pulse_duration, wavelength):
        
        self.pulse_profile = np.loadtxt(pulse_profile, skiprows=1)
        self.pulse_energy = pulse_energy
        self.spot_size = spot_size
        self.pulse_duration = pulse_duration
        self.wavelength = wavelength
        
    def create_power_profile(self):
        
        time = self.pulse_profile[:, 1:2]
        amplitude = self.pulse_profile[:, 2:3]
        
        # Ensure arrays are sorted for plotting
        inds = np.argsort(time.flatten())
        time = time[inds]
        amplitude = amplitude[inds]

        # Normalize amplitude array
        amplitude_norm = amplitude / amplitude.max()

        spot_area_m = np.pi*(self.spot_size/2)**2

        # Create power array
        peak_power = self.pulse_energy / self.pulse_duration
        power_profile = amplitude_norm * peak_power

        peak_energy_density_m = self.pulse_energy / spot_area_m
        energy_density_profile = amplitude_norm * peak_energy_density_m

        peak_power_density_m = peak_power / spot_area_m
        power_density_profile = amplitude_norm * peak_power_density_m
        
        return time, power_profile, energy_density_profile, power_density_profile
    
    def find_drake_pressure(self):
        
        irradiance = (self.pulse_energy / (np.pi * (self.spot_size/2)**2 * self.pulse_duration)) * 1e-4
        
        P_ab = 800 * (irradiance * 1e-14)**(2/3) * (self.wavelength * 1e6)**(-2/3)
        
        return P_ab
    
    def write_pulse_to_outfile(self, time, power_profile, outfile_name):
        
        power_profile = power_profile * 1e-12
        
        with open(f'{outfile_name}.txt', 'w') as f:
            f.write('Laser Drive\n')
            f.write('time (ns), Power (TW)\n')
            f.write('0, 0\n')
            np.savetxt(f, list(zip(time.flatten(), power_profile.flatten())), delimiter=', ', fmt='%f')
            
        print(f'Pulse profile written to {outfile_name}.txt')
        
    def plot_laser_pulse(self, time, power_profile, energy_density_profile, power_density_profile, shot_number):
        
        plt.plot(time, power_profile)
        plt.xlabel('time [ns]')
        plt.ylabel('power [W]')
        plt.title(f'Power vs. Time for Shot {shot_number}')
        plt.show()
        
        plt.plot(time, energy_density_profile)
        plt.xlabel('time [ns]')
        plt.ylabel('energy density [J/m$^2$]')
        plt.title(f'Energy Density vs. Time for Shot {shot_number}')
        plt.show()
        
        plt.plot(time, power_density_profile)
        plt.xlabel('time [ns]')
        plt.ylabel('power density [W/m$^2$]')
        plt.title(f'Power Density vs. Time for Shot {shot_number}')
        plt.show()
        
        plt.plot(time, power_density_profile/power_density_profile.max())
        plt.xlabel('time [ns]')
        plt.ylabel('amplitude [arb. units]')
        plt.title('Laser Profile for Shot 918')
        plt.show()
        
    def average_several_pulses(self):
        
        # This code works under the assumption that each time array is the same
        # Could use interpolation to generalize
        
        pulse_profile1 = 'trace_917.dat'
        pulse_profile2 = 'trace_918.dat'
        averaged_pulse_outfile = 'trace_average.dat'
        
        pulse1 = np.loadtxt(pulse_profile1, skiprows=1)
        pulse2 = np.loadtxt(pulse_profile2, skiprows=1)
        
        time = pulse1[:, 1:2]
        amplitude1_norm = pulse1[:, 2:3]/pulse1[:, 2:3].max()
        amplitude2_norm = pulse2[:, 2:3]/pulse2[:, 2:3].max()
        
        amplitude_av = (amplitude1_norm+amplitude2_norm)/2
        
        steps = np.arange(0, len(time))
        
        with open(f'{averaged_pulse_outfile}', 'w') as f:
            f.write('step time amplitude\n')
            np.savetxt(f, list(zip(steps.flatten(), time.flatten(), amplitude_av.flatten())), fmt='%f')
       
        print(f'Averaged pulse profile written to {averaged_pulse_outfile}')
        
        
class ParticleVelocityAnalysis:
    
    def __init__(self, Up_input_files):
        
        visar = np.loadtxt(f'{Up_input_files[0]}.txt', dtype='float64')
        sim_trace = np.loadtxt(f'{Up_input_files[1]}.csv', delimiter=',', skiprows=2)
        
        self.sim_trace_time = sim_trace[:, 0:1]
        self.sim_trace_Up = sim_trace[:, 1:]
        self.visar_Up = visar[:, 1:]
        self.visar_time = visar[:, 0:1]
        self.time_shift = self.sim_trace_time[self.sim_trace_Up > 0.3][0] - self.visar_time[self.visar_Up > 0.2][0]
        self.visar_time_shifted = self.visar_time + self.time_shift
        
        
    def plot_traces(self, shot_number, multiplier):
        
        plt.plot(self.visar_time_shifted, self.visar_Up, label='experimental trace')
        plt.plot(self.sim_trace_time, self.sim_trace_Up, label='HYADES trace')
        plt.title(f'Experiment vs. HYADES for Shot {shot_number} (multiplier={multiplier})')
        plt.xlabel('time [ns]')
        plt.ylabel('particle velocity [km/s]')
        plt.legend()
        plt.show()
        
    def find_peak_region(self):
        
        while True:

            start_time = float(input('Start time for peak region analysis: '))
            end_time = float(input('End time for peak region analysis: '))
            
            plt.plot(self.visar_time_shifted, self.visar_Up, label='visar trace')
            plt.axvline(x=start_time, color='red', linestyle='--', label='start time')
            plt.axvline(x=end_time, color='red', linestyle='--', label='end time')
            plt.title('Time Region Determination')
            plt.xlabel('time [ns]')
            plt.ylabel('particle velocity [km/s]')
            plt.legend()
            plt.show()
            
            is_region_accurate = str(input('Is the region correct? [y/n]: '))
            if is_region_accurate == 'n':
                continue
            else:
                break
            
        visar_time_mask = (self.visar_time_shifted<end_time) & (self.visar_time_shifted>start_time)    
        sim_time_mask = (self.sim_trace_time<end_time) & (self.sim_trace_time>start_time)  
        
        plt.plot(self.visar_time_shifted[visar_time_mask], self.visar_Up[visar_time_mask], label='visar trace')
        plt.plot(self.sim_trace_time[sim_time_mask], self.sim_trace_Up[sim_time_mask], label='hyades trace')
        plt.title('VISAR vs. HYADES in Peak Region')
        plt.xlabel('time [ns]')
        plt.ylabel('particle velocity [km/s]')
        plt.legend()
        plt.show()
        
        return visar_time_mask, sim_time_mask
    
            
    def chisq_determination(self, visar_time_mask, sim_time_mask):
        
        # Alternative approach is to mask in peak 90% region 
        visar_Up_peak_region = self.visar_Up[visar_time_mask]
        sim_Up_peak_region = self.sim_trace_Up[sim_time_mask]
        visar_time_peak_region = self.visar_time_shifted[visar_time_mask]
        sim_time_peak_region = self.sim_trace_time[sim_time_mask]
        
        interpolation_function = scipy.interpolate.interp1d(sim_time_peak_region, sim_Up_peak_region, fill_value="extrapolate")
        sim_Up_interpolated = interpolation_function(visar_time_peak_region)
        
        residual_array = ((visar_Up_peak_region - sim_Up_interpolated) ** 2) / sim_Up_interpolated
        chisq = np.sum(residual_array)
        Up_average_sim = np.average(sim_Up_peak_region)
        Up_average_visar = np.average(visar_Up_peak_region)
        
        return chisq, Up_average_sim, Up_average_visar
    
    def conduct_chisq_analysis(self, shot_number):
        
        multiplier = float(input('Value of multiplier: '))
        
        self.plot_traces(shot_number, multiplier)
        visar_time_mask, sim_time_mask = self.find_peak_region()
        chisq, Up_average_sim, Up_average_visar = self.chisq_determination(visar_time_mask, sim_time_mask)
        
        print(f'The value of chi-squared for this multiplier is {chisq}')
        print(f'The average simulation peak velocity for this multiplier is {np.around(Up_average_sim, 3)}km/s')
        
        return multiplier, chisq, Up_average_sim, Up_average_visar
        
    def write_chisq_to_outfile(self, chisq_outfile, multiplier, chisq):
        
        file_exists = os.path.isfile(f'{chisq_outfile}.csv')
        with open(f'{chisq_outfile}.csv', 'a+') as f:
            if not file_exists:
                f.write('Multiplier, Chi Squared\n')
            f.write(f'{multiplier}, {chisq}\n')
            
        print(f'Multiplier and chi-squared values written to {chisq_outfile}.csv')
        
    def write_Up_to_outfile(self, Up_average, multiplier, energy_density_profile, Up_outfile):
        
        file_exists = os.path.isfile(f'{Up_outfile}.csv')
        with open(f'{Up_outfile}.csv', 'a+') as f:
            if not file_exists:
                f.write('Multiplier, Energy Density, LiF Particle Velocity, Unadjusted Energy Density\n')
            f.write(f'{multiplier}, {multiplier*np.max(energy_density_profile)}, {Up_average}, {np.max(energy_density_profile)}\n')
            
        print(f'Multiplier, energy density and particle velocity values written to {Up_outfile}.csv')
        
class PressureAnalysis:
    
    def __init__(self, pressure_trace_file):
        
        pressure_trace = np.loadtxt(f'{pressure_trace_file}.csv', delimiter=',', skiprows=2)
        
        self.time = pressure_trace[:, 0:1]
        self.pressure = pressure_trace[:, 1:]
        
    def find_peak_pressure(self):
        
        plt.plot(self.time, self.pressure, label='pressure trace')
        plt.title('Pressure Time Region Determination (Preliminary)')
        plt.xlabel('time [ns]')
        plt.ylabel('pressure [GPa]')
        plt.legend()
        plt.show()
        
        while True:

            start_time = float(input('Start time for peak pressure region analysis: '))
            end_time = float(input('End time for peak pressure region analysis: '))
    
            plt.plot(self.time, self.pressure, label='pressure trace')
            plt.axvline(x=start_time, color='red', linestyle='--', label='start time')
            plt.axvline(x=end_time, color='red', linestyle='--', label='end time')
            plt.title('Pressure Time Region Determination')
            plt.xlabel('time [ns]')
            plt.ylabel('pressure [GPa]')
            plt.legend()
            plt.show()
            
            is_region_accurate = str(input('Is the region correct? [y/n]: '))
            if is_region_accurate == 'n':
                continue
            else:
                break
       
        time_mask = (self.time<end_time) & (self.time>start_time)    
        pressure_average = np.mean(self.pressure[time_mask])
        
        print(f'The average peak pressure is: {pressure_average}')
        
        return pressure_average
    
    
    def write_pressure_to_outfile(self, pressure_average, energy_density_profile, pressure_outfile):
        
        multiplier = float(input('Value of multiplier: '))
        
        file_exists = os.path.isfile(f'{pressure_outfile}.csv')
        with open(f'{pressure_outfile}.csv', 'a+') as f:
            if not file_exists:
                f.write('Multiplier, Energy Density, Pressure, Unadjusted Energy Density\n')
            f.write(f'{multiplier}, {multiplier*np.max(energy_density_profile)}, {pressure_average}, {np.max(energy_density_profile)}\n')
            
        print(f'Multiplier, energy density and pressure values written to {pressure_outfile}.csv')
        

class MultiSimulationAnalysis:
    
    def __init__(self, chisq_outfile, Up_outfile, pressure_outfile):
        
        self.Up_data_path = f'{Up_outfile}.csv'
        self.pressure_data_path = f'{pressure_outfile}.csv'
        self.chisq_data = np.loadtxt(f'{chisq_outfile}.csv', delimiter=',', skiprows=1)
        self.Up_data = np.loadtxt(f'{Up_outfile}.csv', delimiter=',', skiprows=1)
        self.pressure_data = np.loadtxt(f'{pressure_outfile}.csv', delimiter=',', skiprows=1)
        self.convert_lif_to_kapton_parameters()
        
    def convert_lif_to_kapton_parameters(self):
        
        lif_Ups = self.Up_data[:, 2:3].flatten()
        lif_Ups_errors = self.Up_data[:, 7:8]
        
        kapton_Ups = 1.5664 * lif_Ups - 0.061245 * lif_Ups**2
        kapton_pressures = 4.241 * kapton_Ups + 1.4206 * kapton_Ups**2 + 0.040424 * kapton_Ups**3
        
        kapton_Ups_errors = 1.5664 * lif_Ups_errors - 0.061245 * lif_Ups_errors**2
        kapton_pressures_errors = 4.241 * kapton_Ups_errors + 1.4206 * kapton_Ups_errors**2 + 0.040424 * kapton_Ups_errors**3
        
        df_p = pd.read_csv(self.Up_data_path)
        
        df_p['Kapton Particle Velocity'] = kapton_Ups
        df_p['Kapton Pressure'] = kapton_pressures
        
        df_p['Delta Kapton Particle Velocity'] = kapton_Ups_errors
        df_p['Delta Kapton Pressure'] = kapton_pressures_errors
        
        df_p.to_csv(self.Up_data_path, index=False)
    
    def best_multiplier_determination(self, shot_number):
        
        multiplier = self.chisq_data[:, 0:1].flatten()
        chisq_vals = self.chisq_data[:, 1:2].flatten()
        
        # Remove repeated elements
        _, non_repeats = np.unique(multiplier, return_index=True)
        multiplier = multiplier[non_repeats]
        chisq_vals = chisq_vals[non_repeats]
        
        # Set strictly increasing order
        inds = np.argsort(multiplier)
        multiplier_sort = multiplier[inds]
        chisq_vals_sort = chisq_vals[inds]
        
        interpolation_function = scipy.interpolate.CubicSpline(multiplier_sort, chisq_vals_sort)
        x_vals = np.linspace(0.2,1,100)
        interpolated_chisq = interpolation_function(x_vals)
        
        min_ind = np.argmin(interpolated_chisq)
        best_multiplier = np.around(x_vals[min_ind], 2)
        
        filtered_x = x_vals[min_ind-10:min_ind+10]
        filtered_y = interpolated_chisq[min_ind-10:min_ind+10]
        
        coefficients = np.polyfit(filtered_x, filtered_y, 2)
        
        multiplier_error = 1/np.sqrt(coefficients[0])
        
        plt.plot(x_vals[x_vals>0.4], interpolated_chisq[x_vals>0.4], linestyle='--', color='k')
        plt.scatter(multiplier_sort, chisq_vals_sort, marker='s')
        plt.title(f'$\chi^2$ vs. Energy Multiplier for Shot {shot_number}')
        plt.xlabel('Energy Multiplier')
        plt.ylabel('$\chi^2$')
        plt.show()
        
        print(f'The best multiplier is {best_multiplier} +/- {multiplier_error}')
        
        return best_multiplier, multiplier_error
    

    def plot_multiplier(self, spot_size, pulse_duration, wavelength):
        
        cu_data = pd.read_csv('CuData.csv')
        
        which_param = str(input('Are you plotting against pressure or particle velocity? [p/u]: '))
        
        if which_param == 'u':
            
            title_token = 'Ablator Particle Velocity'
            axis_token = 'ablator particle velocity [km/s]'
            param = self.Up_data[:, 4:5].flatten()
            param_errors = self.Up_data[:, 8:9].flatten()
            
            
        elif which_param == 'p':
            
            title_token = 'Ablator Pressure'
            axis_token = 'ablator pressure [GPa]'
            param = self.Up_data[:, 5:6].flatten()
            param_errors = self.Up_data[:, 9:10].flatten()

        else:
            raise ValueError('Invalid input. Select either p or u.')
            
        multiplier = self.Up_data[:, 0:1].flatten()
        energy_density = self.Up_data[:, 1:2].flatten()
        unadjusted_energy_density = self.Up_data[:, 3:4].flatten()
        multiplier_errors = self.Up_data[:, 6:7].flatten()
            
        inds = np.argsort(multiplier)
        
        param = param[inds]
        multiplier = multiplier[inds]
        energy_density = energy_density[inds]
        unadjusted_energy_density = unadjusted_energy_density[inds]
        multiplier_errors = multiplier_errors[inds]
        
        spot_area = np.pi*(spot_size/2)**2
        
        unadjusted_energy = unadjusted_energy_density * spot_area
        energy = energy_density * spot_area
        energy_loss = (unadjusted_energy_density - energy_density)*spot_area
        energy_errors = unadjusted_energy * multiplier_errors
        
        unadjusted_energy_errors = [0, 0, 0, 0, 0, 0.1, 0, 0.1, 0.1, 0.1, 0.1, 0]
        
        cu_energy_loss = cu_data['Energy'] - cu_data['Adjusted Energy']
        
        # linear_regression(proportional=True) forces 0,0 intercept
        
        if which_param == 'p':
        
            slope_multiplier, intercept_multiplier = linear_regression(np.concatenate((param, cu_data['Pressure'])), np.concatenate((multiplier, cu_data['Multiplier'])))
            slope_Eloss, intercept_Eloss, r_value_Eloss, p_value_Eloss, std_err_Eloss = stats.linregress(np.concatenate((param, cu_data['Pressure'])), np.concatenate((energy_loss, cu_energy_loss)))
            
            y_vals_multiplier = slope_multiplier*param+intercept_multiplier
            y_vals_Eloss = slope_Eloss*param+intercept_Eloss
            
        else:
            
            slope_multiplier, intercept_multiplier = linear_regression(param, multiplier)
            slope_Eloss, intercept_Eloss, r_value_Eloss, p_value_Eloss, std_err_Eloss = stats.linregress(param, energy_loss)
            
            y_vals_multiplier = slope_multiplier*param+intercept_multiplier
            y_vals_Eloss = slope_Eloss*param+intercept_Eloss
            
            
        if which_param == 'u':
            
            def objective_Up(x, a, b):
                return a*x**b
            
            popt_unadjusted, _ = scipy.optimize.curve_fit(objective_Up, unadjusted_energy, param)
            a1, b1 = popt_unadjusted
            
            x_unadjusted = np.linspace(0, 40, 100)
            fit_Up_unadjusted_energy = objective_Up(x_unadjusted, a1, b1)
            
            popt_adjusted, _ = scipy.optimize.curve_fit(objective_Up, energy, param)
            a2, b2 = popt_adjusted
            
            x_adjusted = np.linspace(0, 40, 100)
            fit_Up_energy = objective_Up(x_adjusted, a2, b2)
            
            plt.plot(x_adjusted, fit_Up_energy, color='k', linestyle='-.', label='adjusted fit')
            plt.plot(x_unadjusted, fit_Up_unadjusted_energy, color='k', linestyle='--', label='unadjusted fit')
            plt.errorbar(energy, param, xerr=energy_errors, yerr=param_errors, fmt='o', marker='s', label='adjusted energy')
            plt.errorbar(unadjusted_energy, param, xerr=unadjusted_energy_errors, yerr=param_errors, fmt='o', marker='s', label='unadjusted energy')
            plt.xlim(0, max(unadjusted_energy)+5)
            plt.ylim(0, max(param)+0.5)
            plt.title(f'{title_token} vs. Pulse Energy')
            plt.xlabel('pulse energy [J]')
            plt.ylabel(f'{axis_token}')
            plt.legend()
            plt.show()

        if which_param == 'p':

            def objective_p_energy(x, a, b, c):
                
                return a * (x*1e-18/(np.pi*(spot_size/2)**2*pulse_duration))**b * (wavelength*1e6)**c
                
            popt_unadjusted, _ = scipy.optimize.curve_fit(objective_p_energy, np.concatenate((unadjusted_energy, cu_data['Energy'])), np.concatenate((param, cu_data['Pressure'])))
            a1, b1, c1 = popt_unadjusted
        
            x_unadjusted = np.linspace(0, np.max(unadjusted_energy), 100)
            fit_p_unadjusted_energy = objective_p_energy(x_unadjusted, a1, b1, c1)
        
            popt_adjusted, _ = scipy.optimize.curve_fit(objective_p_energy, np.concatenate((energy, cu_data['Adjusted Energy'])), np.concatenate((param, cu_data['Pressure'])))
            a2, b2, c2 = popt_adjusted
            
            x_adjusted = np.linspace(0, np.max(energy), 100)
            fit_p_adjusted_energy = objective_p_energy(x_adjusted, a2, b2, c2)
            
            drake_fit = objective_p_energy(x_unadjusted, 800, 2/3, -2/3)
            
      
            plt.errorbar(cu_data['Adjusted Energy'], cu_data['Pressure'], fmt='o', marker='s', xerr=cu_data['Delta Adjusted Energy'], yerr=cu_data['Delta Pressure'], label='polyimide-Cu')
            #plt.plot(x_unadjusted, drake_fit, color='k', label='ablation scaling law (Drake)')
            plt.errorbar(energy, param, xerr = energy_errors, yerr=param_errors, fmt='o', marker='s', label='polyimide-LiF')
            plt.plot(x_adjusted, fit_p_adjusted_energy, color='k', linestyle='-.', label='adjusted energy fit')
            plt.xlim(0, max(unadjusted_energy)+10)
            plt.ylim(0, max(param)+2)
            plt.title(f'{title_token} vs. Adjusted Energy')
            plt.xlabel('pulse energy [J]')
            plt.ylabel(f'{axis_token}')
            plt.legend(loc='lower right')
            plt.show()
          

            plt.errorbar(cu_data['Energy'], cu_data['Pressure'], fmt='o', marker='s', xerr=cu_data['Delta Energy'], yerr=cu_data['Delta Pressure'], label='polyimide-Cu')
            #plt.plot(x_unadjusted, drake_fit, color='k', label='ablation scaling law (Drake)')      
            plt.errorbar(unadjusted_energy, param, xerr=unadjusted_energy_errors, yerr=param_errors, fmt='o', marker='s', label='polyimide-LiF')
            plt.plot(x_unadjusted, fit_p_unadjusted_energy, color='k', linestyle='--', label='unadjusted energy fit')
            plt.xlim(0, max(unadjusted_energy)+10)
            plt.ylim(0, max(param)+2)
            plt.title(f'{title_token} vs. Unadjusted Energy')
            plt.xlabel('pulse energy [J]')
            plt.ylabel(f'{axis_token}')
            plt.legend(loc='lower right')
            plt.show()
                
        
        if which_param == 'p':
            plt.errorbar(cu_data['Pressure'], cu_data['Multiplier'], yerr=cu_data['Delta Multiplier'], xerr=cu_data['Delta Pressure'], fmt='o', marker='s', label='polyimide-Cu 500$\mu$m PP')
            plt.xlim(0, max(param)+5)
            
        elif which_param == 'u':
            plt.xlim(0, max(param)+0.5)
            
        plt.plot([param[0], max(param)], [y_vals_multiplier[0], max(y_vals_multiplier)], color='k', linestyle='-.')
        plt.errorbar(param, multiplier, yerr=multiplier_errors, xerr=param_errors, fmt= 'o', marker='s', label='polyimide-LiF 500$\mu$m PP')
        plt.ylim(0, max(multiplier)+0.2)
        plt.title(f'Energy Multiplier vs. {title_token}')
        plt.xlabel(f'{axis_token}')
        plt.ylabel('multiplier')
        plt.legend(loc='best')
        plt.show()
        
        if which_param == 'p':
            plt.errorbar(cu_data['Pressure'], cu_energy_loss, yerr=cu_data['Delta Adjusted Energy'], xerr=cu_data['Delta Pressure'], fmt='o', marker='s', label='polyimide-Cu 500$\mu$m PP')
            plt.xlim(0, max(param)+5)   
        elif which_param == 'u':
            plt.xlim(0, max(param)+0.5)
            
        plt.plot([param[0], max(param)], [y_vals_Eloss[0], max(y_vals_Eloss)], color='k', linestyle='-.')
        plt.errorbar(param, energy_loss, yerr=energy_errors, xerr=param_errors, fmt='o', marker='s', label='polyimide-LiF 500$\mu$m PP')
        plt.ylim(0, max(energy_loss)+2)
        plt.title(f'Energy Loss vs. {title_token}')
        plt.xlabel(f'{axis_token}')
        plt.ylabel('energy loss [J]')
        plt.legend(loc='best')
        plt.show()
        
        
        if which_param == 'p':
            plt.errorbar(cu_data['Pressure'], (1-cu_data['Multiplier'])*100, xerr=cu_data['Delta Pressure'], yerr=cu_data['Delta Multiplier']*100, fmt='o', marker='s', label='polyimide-Cu')
            plt.xlim(0, max(param)+5)
        elif which_param == 'u':
            plt.xlim(0, max(param)+0.5)
            
        plt.plot([param[0], max(param)], [((1-y_vals_multiplier)*100)[0], min((1-y_vals_multiplier)*100)], color='k', linestyle='-.')
        plt.errorbar(param, (1-multiplier)*100, xerr=param_errors, yerr=multiplier_errors*100, fmt='o', marker='s', label='polyimide-LiF')
        plt.ylim(0, 80)
        plt.title(f'Percentage Energy Loss vs. {title_token}')
        plt.xlabel(f'{axis_token}')
        plt.ylabel('% energy loss')
        plt.legend(loc='best')
        plt.show()
        
        slope_Eloss, intercept_Eloss, r_value_Eloss, p_value_Eloss, std_err_Eloss = stats.linregress(np.concatenate((unadjusted_energy, cu_data['Energy'])), np.concatenate((energy_loss, cu_energy_loss)))
        y_vals_Eloss = slope_Eloss*unadjusted_energy+intercept_Eloss
        
        plt.plot([unadjusted_energy[0], max(unadjusted_energy)], [y_vals_Eloss[0], max(y_vals_Eloss)], color='k', linestyle='-.')
        plt.errorbar(cu_data['Energy'], cu_energy_loss, yerr=cu_data['Delta Adjusted Energy'], xerr=cu_data['Delta Energy'], fmt='o', marker='s', label='polyimide-Cu 500$\mu$m PP')
        plt.errorbar(unadjusted_energy, energy_loss, xerr=unadjusted_energy_errors, yerr=energy_errors, fmt='o', marker='s', label='polyimide-LiF 500$\mu$m PP')
        plt.ylim(0, max(energy_loss)+2)
        plt.title(f'Energy Loss vs. Pulse Energy')
        plt.xlabel(f'pulse energy [J]')
        plt.ylabel('energy loss [J]')
        plt.legend(loc='best')
        plt.show()
        
        slope_multiplier, intercept_multiplier = linear_regression(np.concatenate((unadjusted_energy, cu_data['Energy'])), np.concatenate((multiplier, cu_data['Multiplier'])))
        y_vals_multiplier = slope_multiplier*unadjusted_energy+intercept_multiplier
        
        plt.plot([unadjusted_energy[0], max(unadjusted_energy)], [((1-y_vals_multiplier)*100)[0], min((1-y_vals_multiplier)*100)], color='k', linestyle='-.')
        plt.errorbar(cu_data['Energy'], (1-cu_data['Multiplier'])*100, xerr=cu_data['Delta Energy'], yerr=cu_data['Delta Multiplier']*100, fmt='o', marker='s', label='polyimide-Cu 500$\mu$m PP')
        plt.errorbar(unadjusted_energy, (1-multiplier)*100, xerr=unadjusted_energy_errors, yerr=multiplier_errors*100, fmt='o', marker='s', label='polyimide-LiF 500$\mu$m PP')
        plt.ylim(0, 80)
        plt.title(f'Percentage Energy Loss vs. Pulse Energy')
        plt.xlabel('pulse energy [J]')
        plt.ylabel('% energy loss')
        plt.legend(loc='best')
        plt.show()
        

    def compare_Up_traces(self, input_deck, labels, visar_labels, title):
        
        # Change linestyles and colours depending on size of input deck
        linestyles = np.array(['--', '--', '--', '--'])
        colours = np.array(['b', 'g', 'r', 'm'])
        
        objects = []
        
        for item in input_deck:
            
            objects.append(ParticleVelocityAnalysis(item))
            
        plt.plot(objects[0].sim_trace_time, objects[0].sim_trace_Up)
        plt.title('Preliminary Plot')
        plt.xlabel('time [ns]')
        plt.ylabel('particle velocity [km/s]')
        plt.show()
        
        while True:
        
            start_time = float(input('Start time for peak pressure region analysis: '))
            end_time = float(input('End time for peak pressure region analysis: '))
            
            plt.plot(objects[0].visar_time_shifted, objects[0].visar_Up)
            plt.axvline(x=start_time, color='red', linestyle='--', label='start time')
            plt.axvline(x=end_time, color='red', linestyle='--', label='end time')
            plt.title('Time Region Determination')
            plt.xlabel('time [ns]')
            plt.ylabel('particle velocity [km/s]')
            plt.legend()
            plt.show()
            
            is_region_accurate = str(input('Is the region correct? [y/n]: '))
            if is_region_accurate == 'n':
                continue
            else:
                break
            
        peak_Ups = []
            
        shift_reference = objects[0].sim_trace_time[objects[0].sim_trace_Up>0.4][0]
        
        
        visar_num = str(input('Plot only 1 VISAR? [y/n]: '))
            
        if visar_num == 'y':
        
            initial_time_mask = ((objects[0].visar_time_shifted<end_time) & (objects[0].visar_time_shifted>start_time))
            visar_peak_Up = np.average(objects[0].visar_Up[initial_time_mask])
            plt.plot(objects[0].visar_time_shifted[initial_time_mask], objects[0].visar_Up[initial_time_mask], color='k', linestyle='-', label='VISAR', lw=0.9)
        
        for obj, lbl, visar_lbl, linestyle, colour in zip(objects, labels, visar_labels, linestyles, colours):
            
            shift = shift_reference - obj.sim_trace_time[obj.sim_trace_Up>0.2][0]
            time_mask = ((obj.sim_trace_time+shift)<end_time) & (obj.sim_trace_time+shift>start_time)
            peak_Ups.append(np.average(obj.sim_trace_Up[time_mask]))
            time_mask_visar = ((obj.visar_time_shifted+shift)<end_time) & (obj.visar_time_shifted+shift>start_time)
            plt.plot(obj.sim_trace_time[time_mask]+shift, obj.sim_trace_Up[time_mask], linestyle=linestyle, color=colour, label=lbl, lw=0.9)
            
            if visar_num == 'n':
                
                plt.plot(obj.visar_time_shifted[time_mask_visar]+shift, obj.visar_Up[time_mask_visar], label=visar_lbl, color='k', lw=0.9)
        
        plt.xlabel('time [ns]')
        plt.ylabel('particle velocity [km/s]')
        plt.title(title)
        plt.legend()
        plt.show()
        
        return peak_Ups
        
    def find_bias_due_to_epoxy(self, input_deck, Up_labels, visar_labels, title):
            
        peak_Ups = self.compare_Up_traces(input_deck, Up_labels, visar_labels, title)
        
        no_epoxy_Up = peak_Ups[0]
        Up_average = np.average(peak_Ups[1:])
        average_bias = no_epoxy_Up - Up_average
        bias_precision = np.std(peak_Ups[1:])
        
        percent_bias = average_bias/no_epoxy_Up
        
        print(f'The percent error in peak particle velocity due to non-consideration of an epoxy layer is: {percent_bias}')
        print(f'The precision in the bias is: {bias_precision}')
        
        # no epoxy requires a lower multiplier for best fit (on average)
        # therefore the energy loss is slightly overestimated
        # precision in bias > bias, therefore insignificant
        
    
            
def main():
    
    """
    To generate laser pulse, change laser input and output file names, and pulse parameters
    To conduct analysis change visar/simulation input and chi squared output file names
    """
    
    # Pulse parameters
    pulse_profile = 'trace_918.dat'
    pulse_energy = 30
    spot_size = 500e-6
    pulse_duration = 10e-9
    wavelength = 515e-9
    
    laser_outfile = 'convergence_pulse'
    
    # Shot number parameter is only used in plot titles
    # Could change this to be more efficient using re functions
    shot_number = 157
    chisq_outfile = 'dipole171_chisq'
    
    # Analysis files
    visar_file = '157_visar_18p90J'
    sim_trace_file = 'dipole157_067_ParticleVelocity_50.0um'
    Up_input_files = np.array([visar_file, sim_trace_file])
    
    pressure_trace_file = 'Cu38p40_060_Pressure_15.0um'
    
    # Multi-analysis files
    Up_outfile = 'Up_analysis'
    pressure_outfile = 'Cu38p40_pressure_analysis' # can be empty as is now not used in multi_analysis
    
    #Up_input_deck = np.array([['163_visar_4.729J', 'dipole163_046_ParticleVelocity_50.0um'],\
    #                       ['171_visar_9p580J', 'dipole171_066_ParticleVelocity_50.0um'],                             
    #                       ['168_visar_26p9J', 'dipole168_082_ParticleVelocity_50.0um'],                           
    #                       ['918_time_Up', 'dipole918_079_ParticleVelocity_50.0um']])
    
    Up_input_deck = np.array([['171_visar_9p580J', 'dipole171_066_ParticleVelocity_50.0um'],\
                           ['171_visar_9p580J', 'dipole171_epoxy1_ParticleVelocity_51.0um'],
                           ['171_visar_9p580J', 'dipole171_epoxy3_ParticleVelocity_53.0um'],
                           ['171_visar_9p580J', 'dipole171_epoxy5_ParticleVelocity_55.0um']])
        
    Up_labels = np.array(['no epoxy', '1$\mu$m epoxy', '3$\mu$m epoxy', '5$\mu$m epoxy'])
                    
    visar_labels = np.array(['171 VISAR', '171 VISAR', '171 VISAR', '171 VISAR'])
    
    Up_title = 'Particle Velocity vs. Time For Shot 171'
    
    
    #laser_pulse = LaserPulse(pulse_profile, pulse_energy, spot_size, pulse_duration, wavelength)
    #time, power_profile, energy_density_profile, power_density_profile = laser_pulse.create_power_profile()
    #laser_pulse.write_pulse_to_outfile(time, power_profile, laser_outfile)
    #laser_pulse.plot_laser_pulse(time, power_profile, energy_density_profile, power_density_profile, shot_number)
    #laser_pulse.average_several_pulses()
    #laser_pulse.find_drake_pressure()
    
    analysis = ParticleVelocityAnalysis(Up_input_files)
    multiplier, chisq, Up_average_sim, Up_average_visar = analysis.conduct_chisq_analysis(shot_number)
    #analysis.write_chisq_to_outfile(chisq_outfile, multiplier, chisq)
    #analysis.write_Up_to_outfile(Up_average_sim, multiplier, energy_density_profile, Up_outfile)
    
    #pressure_analysis = PressureAnalysis(pressure_trace_file)
    #pressure_av = pressure_analysis.find_peak_pressure()
    #pressure_analysis.write_pressure_to_outfile(pressure_av, energy_density_profile, pressure_outfile)
    
    #multi_analysis = MultiSimulationAnalysis(chisq_outfile, Up_outfile, pressure_outfile)
    #multi_analysis.best_multiplier_determination(shot_number)
    #multi_analysis.plot_multiplier(spot_size, pulse_duration, wavelength)
    #multi_analysis.compare_Up_traces(Up_input_deck, Up_labels, visar_labels, Up_title)
    #multi_analysis.find_bias_due_to_epoxy(Up_input_deck, Up_labels, visar_labels, Up_title)
    

if __name__ == "__main__":
    main()
    

