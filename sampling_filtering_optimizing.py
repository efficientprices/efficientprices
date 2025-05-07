import pandas as pd
import numpy as np
from datetime import timedelta, datetime, date
import math
from scipy.optimize import minimize
from casadi import *
import time
import os
import json


class Filter():
    """
    Class that implements the Kalman filter for given sampled data or also tick data.
    The class is used to filter the data and save the filtered data to a CSV file.
    """

    def __init__(self, path, schemes,durations,window_sizes,start_dates,end_dates):
        """
        :param path: Path to find sampled/tick data
        :param schemes: Sampling schemes for which the filtering step is run ["bts","cts","tts","tt"]
        :param durations: Frequencies considered in the filtering step provided as average interval length between observations in seconds [60,180,300]
        :param window_sizes: dictionary with frequencies as keys and list of window sizes as entries
        :param start_dates: dictionary with frequencies as keys and start dates as entries
        :param end_dates: dictionary with frequencies as keys and end dates as entries
        """
        # Path to store dataframes in
        self.path = path
        self.schemes = schemes
        self.durations = durations
        self.window_sizes = window_sizes
        self.start_dates = start_dates
        self.end_dates = end_dates
        self.start_date = None
        self.end_date = None
        self.observations = 0
        self.estimate_dict = {}
        # build empty dictionary
        for duration in self.durations:
            self.estimate_dict[str(duration)] = {}
            for scheme in self.schemes:
                self.estimate_dict[str(duration)][str(scheme)] = []

    def run_Kalman_filter(self, save_estimates=False,save_estimates_path = None):
        """
        Run the Kalman filter for each scheme and duration, and save the estimates to a JSON file.
        :param save_estimates: Boolean whether the MLE estimates should be saved
        :param save_estimates_path: Path to save the estimates
        """
        for duration in self.durations:
            self.start_date = self.start_dates[duration]
            self.end_date = self.end_dates[duration]
            for scheme in self.schemes:
                print("Filerting " +str(scheme)+" for duration " + str(duration) + " seconds")
                self.moving_window_size(duration, scheme, self.window_sizes[duration], self.path)
        if save_estimates:
            # Write the dictionary to a JSON file
            with open(save_estimates_path, 'w') as json_file:
                    json.dump(self.estimate_dict, json_file)
        self.merge_window_datanew(self.path, self.schemes, self.durations)

    def MEL_function(self, parameters):
        """
        Function that returns the maxmimum likelihood function used for the estimation of the two variances
        param parameters: Current guesses of the two variances that need to be estimated
        return log_likely :Log-likelihood functino to be minimized
        """
        var_epsilon, var_eta = parameters
        N = len(self.observations)
        x_N, K_N, u_N, F_N, P_N = self.kalman_filter_iterativ(N, self.observations, var_epsilon, var_eta)
        log_likely = -(N / 2) * np.log(2 * math.pi) - (1 / 2) * self.sum_like(F_N, u_N)
        return -log_likely

    def sum_like(self, F_N, u_N):
        """
        Help function used for computing log-likelihood function
        Input:
        F_N
        u_N
        Output:
        Expression needed in the log-likelihood function
        """
        output = 0
        for F_i, u_i in zip(F_N, u_N):
            value = np.log(F_i) + (u_i ** 2) / F_i
            output += value
        return output

    def kalman_filter_iterativ(self, N, observations, var_epsilon, var_eta):
        """
        Function that runs the kalman filter for all given observations.
        Input:
        N: Number of observations
        observations: Prices to be filtered
        var_epsilon: MLE estimate of MMN noise variance
        var_eta: MLE estimate of true price variance
        """
        for t in range(N):
            if t == 0:
                x_initial = 0
                P_t_initial = 100000000000000000
                u_t = [observations[t] - x_initial]
                F_t = [P_t_initial + var_epsilon]
                K_t = [P_t_initial / F_t[t]]
                P_t = [P_t_initial * (1 - K_t[t]) + var_eta]
                x_t = [x_initial + K_t[t] * u_t[t]]
            else:
                u_t += [observations[t] - x_t[t - 1]]
                F_t += [P_t[t - 1] + var_epsilon]
                K_t += [P_t[t - 1] / F_t[t]]
                P_t += [P_t[t - 1] * (1 - K_t[t]) + var_eta]
                x_t += [x_t[t - 1] + K_t[t] * u_t[t]]
        return x_t, K_t, u_t, F_t, P_t

    def moving_window_tick_data(self, df_out, window_sizes):
        """
        Function running the Kalman filter for tick data.
        """

        prices = list(df_out["Observations"])
        estimates = {"tick": []}
        for window in window_sizes:
            n_windows = len(prices) // window
            if n_windows == 0:
                n_windows = 1
            prices_window = []
            for n in range(n_windows):
                if n == n_windows - 1:
                    prices_window += [prices[n * window:]]
                else:
                    prices_window += [prices[n * window:(n + 1) * window]]
            filtered_window = []
            for intervall in prices_window:
                log_price = intervall
                # global observations
                self.observations = log_price
                mle_estimates = minimize(self.MEL_function, np.array([0.0000001, 0.0000001]),
                                         bounds=((-np.inf, np.inf), (0, np.inf)), method='Nelder-Mead')
                print(mle_estimates.success)
                if mle_estimates.success == True:
                    var_epsilon_MLE = mle_estimates.x[0]
                    var_price_MLE = mle_estimates.x[1]
                    print(var_epsilon_MLE)
                    print(var_price_MLE)
                    if var_epsilon_MLE < 0 or var_price_MLE < 0:
                        print("Warning negative variance!!!!!!!")
                        var_epsilon_MLE = np.abs(var_epsilon_MLE)
                        var_price_MLE = np.abs(var_price_MLE)
                    estimates["tick"].append(np.abs(var_epsilon_MLE))
                    x, K_t, u_t, F_t, P_t = self.kalman_filter_iterativ(len(self.observations), self.observations,
                                                                        var_epsilon_MLE, var_price_MLE)
                    filtered_window += x
                elif mle_estimates.success == False:
                    print("no solution found")
                    filtered_window += self.observations
            string_df = "Filtered prices " + str(window)
            df_out[string_df] = filtered_window
        return df_out

    def moving_window_size(self, duration, scheme, window_sizes, path):
        """
        Function running the Kalman filter for a given set of estimation window sizes. Computes a dataframe with all filtered prices and the mean filtered price for one given scheme and given frequency.

        Input:
        duration: frequency considered in the filtering step provided as average intervall length between observations in seconds [60,180,300]
        scheme: Sampling schemes for which the filtering step is run ["bts","cts","tts","tt"]
        window_sizes: array consisting of window sizes in number of observatins, e.g. for 300 seconds [79,790,1580,4740,9480] corresponding to 1day, 2weeks, 1month,3months, 6months
        path: Path to store filtered data at

        Output:
        dataframe: Consisting of columns with filtered prices each with a different window size and one column that summarizes the average filtered price. First three columns describe the date, the seconds since opening of the stock market, and the raw sampled un-filtered observations.
        """
        df_out = pd.DataFrame()
        # load observed prices
        df_raw = pd.read_csv("HF_Data/IBM/resampled_prices/" + str(duration) + "_seconds/" + scheme + ".csv",
                             header=None)
        df_raw = df_raw.set_index(0)
        df_raw = df_raw.loc[self.start_date:self.end_date]
        df_raw = df_raw.reset_index()
        prices = self.calculate_log_price(df_raw)
        df_out["Date"] = df_raw.iloc[:, 0]
        df_out["Seconds"] = df_raw.iloc[:, 1]
        df_out["Observations"] = list(prices.iloc[:, 0])
        prices = list(prices.iloc[:, 0])
        for window in window_sizes:
            # print(window)
            n_windows = len(prices) // window
            if n_windows == 0:
                n_windows = 1
            prices_window = []
            for n in range(n_windows):
                if n == n_windows - 1:
                    prices_window += [prices[n * window:]]
                else:
                    prices_window += [prices[n * window:(n + 1) * window]]
            filtered_window = []
            for intervall in prices_window:
                log_price = intervall
                # global observations
                self.observations = log_price
                mle_estimates = minimize(self.MEL_function, np.array([0.0000001, 0.0000001]),
                                         bounds=((-np.inf, np.inf), (0, np.inf)), method='Nelder-Mead')
                print(mle_estimates.success)
                if mle_estimates.success == True:
                    var_epsilon_MLE = mle_estimates.x[0]
                    var_price_MLE = mle_estimates.x[1]
                    print(var_epsilon_MLE)
                    print(var_price_MLE)
                    if var_epsilon_MLE < 0 or var_price_MLE < 0:
                        print("Warning negative variance!!!!!!!")
                        var_epsilon_MLE = np.abs(var_epsilon_MLE)
                        var_price_MLE = np.abs(var_price_MLE)
                    self.estimate_dict[str(duration)][str(scheme)].append(np.abs(var_epsilon_MLE))
                    x, K_t, u_t, F_t, P_t = self.kalman_filter_iterativ(len(self.observations), self.observations,
                                                                        var_epsilon_MLE, var_price_MLE)
                    filtered_window += x
                elif mle_estimates.success == False:
                    print("no solution found")
                    filtered_window += self.observations
            string_df = "Filtered prices " + str(window)
            df_out[string_df] = filtered_window
        if not os.path.exists(path + str(duration) + "_seconds/"):
            # Create the directory
            os.makedirs(path + str(duration) + "_seconds/")
        df_out.to_csv(path + str(duration) + "_seconds/moving_window_kalman_filter_" + str(scheme) + "_new.csv")

    def calculate_log_price(self,raw_data):
        """
        Calculates the log-prices given the resampled data.
        :param: raw_data: Dataframe with the resampled data
        """
        data=[]
        for row in range(raw_data.shape[0]):
            date=raw_data.iloc[row,0]
            log_price=np.log(raw_data.iloc[row,2])
            line=[date, log_price]
            data+=[line]
        df_log_price=pd.DataFrame(data,columns=["Date", "log-Price"])
        df_log_price=df_log_price.set_index(["Date"])
        return df_log_price

    def calculate_timestamp(self, raw_data):
        """
        Calculates datetime object given the date and seconds per midnight.

        Input:
        Dataframe: Having two columns, date and seconds since midnight

        Output:
        data: Array of datetime objects
        """
        data = []
        timestamp = 34200
        for row in range(raw_data.shape[0]):
            date = raw_data.iloc[row, 1]
            timestamp = 34200 + raw_data.iloc[row, 2]
            hours = int(timestamp // 3600)
            minutes = int((timestamp % 3600) // 60)
            seconds = int(((timestamp % 3600) % 60) // 1)
            microseconds = int((((timestamp % 3600) % 60) % 1) * 1000000)
            year = int(str(date)[0:4])
            month = int(str(date)[4:6])
            day = int(str(date)[6:8])
            time = datetime(year=year, month=month, day=day, hour=hours, minute=minutes, second=seconds,
                            microsecond=microseconds)
            str_time = time.isoformat()
            final_str = str_time.replace("T", " ")
            line = [final_str]
            data += [line]
        return data

    def merge_window_datanew(self, path, schemes, avg_dur):
        """
        This function merges the prices after the filtering step into one dataframe.

        Input:
        path: Path to find dataframe with all filtered prices of all windows
        schemes: Sampling schemes for which the filtering step is run ["bts","cts","tts","tt"]
        avg_dur: Frequencies considered in the filtering step provided as average interval length between observations in seconds [60,180,300]

        Output:
        Dataframe: Consisting of all filtered prices among all frequencies, schemes, and Kalman filter window sizes
        """
        switch = 0
        for scheme in schemes:
            for duration in avg_dur:
                print(str(scheme) + str(duration))
                df_window = pd.read_csv(
                    self.path + str(duration) + "_seconds/moving_window_kalman_filter_" + str(scheme) + ".csv")

                # Using "Date" and "Seconds" columns as the index
                df_window.set_index(["Date", "Seconds"], inplace=True)
                true_prices = df_window.iloc[:, 2:]  # Assuming remaining columns start from index 2

                if switch == 0:
                    switch = 1
                    df_all = true_prices.copy()
                    old_suffix = (scheme, str(duration))
                else:
                    df_join = true_prices.copy()
                    df_all = df_all.merge(df_join, left_index=True, right_index=True, how='outer',
                                          suffixes=('old_' + old_suffix[1] + '_' + old_suffix[0],
                                                    '_' + str(duration) + '_' + scheme))
                    old_suffix = (scheme, str(duration))

        df_all = df_all.astype(float)
        df_all.to_csv(path + "all_filtered_prices.csv")


class Optimize_Price():
    """
    Class that implements the non-linear optimization problem to extract efficient prices from a set of filtered prices.
    """
    def __init__(self, T, filtered_prices, daily_obs, mean_returns=0, n_lag=10):
        """
        :param T: number of time steps
        :param filtered_prices: array of T arrays with filtered prices
        :param daily_obs: number of daily time steps
        :param mean_returns: initial mean returns
        :param n_lag: number of considered autocorrelation lags
        """
        # number of time_steps
        self.T = T
        # number of daily time_steps
        self.daily_obs = daily_obs
        # filtered prices -> array of T arrays with filtered prices
        self.filtered_prices = filtered_prices
        # number of considered autocorrelation lags
        self.n_lag = n_lag
        # Initialization container Decision Variables, Lower Bound, Upper Bound
        self.decision_variables_container = {'decision_variable': [], 'lower_bound': [], 'upper_bound': [], 'x_0': []}
        # Initialization container Constraints, Lower Bound, Upper Bound
        self.constraints_container = {'constraint': [], 'lower_bound': [], 'upper_bound': []}
        # Initialization cost function
        self.objective = 0
        # Initialization dictionary for decision variables; key: symbolic name; entry: casadi SX.variable
        self.decision_variables_dictionary = {}
        # returns dictionary
        self.returns_dict = {}
        # Zähler of Cost function
        self.objective_numerator = 0
        # Nenner of Cost function
        self.objective_denominator = 0
        # Sum of all returns
        self.sum_returns = 0
        # x0 mean returns
        self.x0_mean_returns = mean_returns

    def run_optimization(self, save_prices=False, save_path=None):
        """
        run_optimization is the main function to run the optimization for a given time horizon

        """
        print("Start Constructing")
        # clear containers
        self.decision_variables_container = {'decision_variable': [], 'lower_bound': [], 'upper_bound': [], 'x_0': []}
        self.constraints_container = {'constraint': [], 'lower_bound': [], 'upper_bound': []}

        # Define decision variables, constraints and cost fucntion for all n time steps
        self.define_sum_decision_variable()
        # for t in range(self.T):
        self.define_decision_variables()
        for t in range(self.T):
            self.define_constraints(t)
        self.define_objective()

        # get constraint for sum average
        self.get_average_price()

        # Add together cost function
        self.get_objective_function()

        print("Start Solving")
        starttime = time.time()
        # Call Solver
        x = self.decision_variables_container['decision_variable']
        g = self.constraints_container['constraint']
        nlp_prob = {'x': vertcat(*x), 'f': self.objective, 'g': vertcat(*g)}
        nlp_solver = nlpsol('nlp_solver', 'ipopt', nlp_prob,
                            {"ipopt.hessian_approximation": "limited-memory", "ipopt.constr_viol_tol": 1e-15,
                             "ipopt.acceptable_tol": 1e-6})
        sol = nlp_solver(x0=self.decision_variables_container['x_0'],
                         lbx=vertcat(self.decision_variables_container['lower_bound']),
                         ubx=vertcat(self.decision_variables_container['upper_bound']),
                         lbg=vertcat(self.constraints_container['lower_bound']),
                         ubg=vertcat(self.constraints_container['upper_bound']))
        print("Optimization Done")
        endtime = time.time()
        elapsed_time = endtime - starttime
        print("Elapsed Time " + str(elapsed_time))
        if save_prices:
            if save_path != None:
                prices = [price[0] for price in sol["x"].full()[1:]]
                df = pd.DataFrame(prices, columns=["true_price"])
                df.to_csv(save_path, index=False)
        return sol["x"]

    def define_sum_decision_variable(self):
        """
        define_sum_decision_variable() defines the variable that captures the sum of all prices
        """
        self.set_decision_variable("r_sum", -np.inf, np.inf, self.x0_mean_returns)

    def define_decision_variables(self):
        """
        define_decision_variables defines all decision variables for the optimization problem

        """
        # Decision Variable for Optimal Price
        counter_returns = 0
        for t in range(self.T):
            self.set_decision_variable("X_" + str(t), -np.inf, np.inf, np.mean(self.filtered_prices[t]))
            if (t - ((t // self.daily_obs) * self.daily_obs)) != 0:
                counter_returns += 1
                self.returns_dict["r_" + str(counter_returns)] = self.decision_variables_dictionary["X_" + str(t)] - \
                                                                 self.decision_variables_dictionary["X_" + str(t - 1)]
                self.sum_returns += self.decision_variables_dictionary["X_" + str(t)] - \
                                    self.decision_variables_dictionary["X_" + str(t - 1)]

    def set_decision_variable(self, decision_variable_name, lower_bound, upper_bound, x0):
        """
        set_decision_variable creates decision variables, bounds and discrete flag

        :param decision_variable_name: decision_variable_name [str]
        :param lower_bound: lower_bound [int]
        :param upper_bound: upper_bound [int]
        """
        # create CASADI symbolic variable
        variable = SX.sym(decision_variable_name)
        # assign to decision variable dictionary
        self.decision_variables_dictionary[decision_variable_name] = variable
        # assign to decision variable container
        self.decision_variables_container['decision_variable'] += [variable]
        self.decision_variables_container['lower_bound'] += [lower_bound]
        self.decision_variables_container['upper_bound'] += [upper_bound]
        self.decision_variables_container['x_0'] += [x0]

    def define_objective(self):
        """
        define_cost_function writes autocorrelation function of problem
        """
        print("Getting Objective")
        # fill numerator
        for lag in range(1, self.n_lag + 1):
            print(str(lag))
            day = -1
            sum_lag = 0
            for t in range(self.T):
                if (t - ((t // self.daily_obs) * self.daily_obs)) == 0:
                    day += 1
                elif t - day > lag:
                    sum_lag += (self.returns_dict["r_" + str(t - day)] - self.decision_variables_dictionary[
                        "r_sum"]) * (self.returns_dict["r_" + str(t - day - lag)] - self.decision_variables_dictionary[
                        "r_sum"])
            self.objective_numerator += sum_lag ** 2

        print("Getting Denominator")
        # fill denominator
        for t in range(self.T):
            if t - ((t // self.daily_obs) * self.daily_obs) != 0:
                self.objective_denominator += (self.decision_variables_dictionary["X_" + str(t)] -
                                               self.decision_variables_dictionary["X_" + str(t - 1)] -
                                               self.decision_variables_dictionary["r_sum"]) ** 2

    def get_objective_function(self):
        """
        get_cost_function combines numerator and denominator of cost function
        """
        self.objective = (self.objective_numerator) / (self.objective_denominator)

    def get_average_price(self):
        """
        get_average_price defines the average return variable r_sum
        """

        self.constraints_container['constraint'] += [
            self.decision_variables_dictionary["r_sum"] - (1 / (self.T - 1)) * self.sum_returns]
        self.constraints_container['lower_bound'] += [0]
        self.constraints_container['upper_bound'] += [0]

    def define_constraints(self, t):
        """
        define_constraints defines the constraints that are later needed for the formulation of an optimization problem

        : param t: current timestep
        """
        # Constraints price range
        self.define_constraints_price_range(t)

    def define_constraints_price_range(self, t):
        """
        define_constraints_price_range defines the price range the optimal price can lie in

        : param t: current timestep
        """

        # get array of filtered prices for current time intervall
        prices = self.filtered_prices[t]

        # get upper and lower bound for prices
        ub = max(prices)
        lb = min(prices)

        # add constraint
        self.constraints_container['constraint'] += [self.decision_variables_dictionary["X_" + str(t)]]
        self.constraints_container['lower_bound'] += [lb]
        self.constraints_container['upper_bound'] += [ub]


