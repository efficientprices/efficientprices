import pandas as pd
import numpy as np
from datetime import timedelta, datetime, date
import math
from scipy.optimize import minimize
from casadi import *
import time
import os
import json
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.stats.diagnostic import acorr_lm, acorr_ljungbox


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

    def run_Kalman_filter(self, path_sampled_data, save_estimates=False,save_estimates_path = None):
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
                self.moving_window_size(path_sampled_data,duration, scheme, self.window_sizes[duration], self.path)
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

    def moving_window_size(self,path_sampled_data ,duration, scheme, window_sizes, path):
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
        df_raw = pd.read_csv(path_sampled_data + str(duration) + "_seconds/" + scheme + ".csv",
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





class Collect_prices():
    """
    Class that collects the filtered prices for the given time period and frequency. Used as a preprocessing stop before
    the optimization step.
    """
    def __init__(self, path_to_csv, start_date, end_date, frequency=300, window=150, path=None):
        """
        :param path_to_csv: path to the csv file containing the filtered prices
        :param start_date: start date of the time period in the format YYYY-MM-DD
        :param end_date: end date of the time period in the format YYYY-MM-DD
        :param frequency: frequency of the data in seconds
        :param window: window size in seconds
        :param path: path to save the filtered prices
        """
        self.path = path
        # data to filtered prices
        self.data = pd.read_csv(path_to_csv)
        # start date
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        # end date
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        # frequency in s
        self.frequency = frequency
        self.daily_obs = 23400 // self.frequency + 1
        # window size in s
        self.window = window
        # price intervalls
        self.prices = []
        # mean prices
        self.mean_prices = []
        # date range
        self.dates = []

    def run_collect_prices(self,sampled=False,tick=False,get_raw=False):
        """
        Runs the functions to collect distribution of filtered prices for each price tick to be optimized
        :param sampled: boolean whether the data is sampled or not
        :param tick: boolean whether the data is tick data or not
        :param get_raw: boolean whether the raw data should be returned or not
        """
        self.prepare_dataframe(sampled)
        self.get_date_range()
        if tick:
            self.get_list_of_prices_tick(raw=get_raw)
        else:
            self.get_list_of_prices()
        return self.prices

    def prepare_dataframe(self, sampled=False):
        """
        Prepares dataframe of all filtered prices for the next steps
        :param sampled: boolean whether the data is sampled data or not
        """
        # convert string index to datetime object
        if sampled:
            self.data["Date"] = pd.to_datetime(self.data['Date'], format='%Y-%m-%d')
        else:
            self.data["Date"] = pd.to_datetime(self.data['Date'], format='%Y%m%d')
            self.data['Date'] = self.data['Date'].dt.strftime('%Y-%m-%d')
        # set date as index
        self.data = self.data.set_index("Date")
        self.data['Seconds'] = self.data['Seconds'].astype(float)

    def get_list_of_prices(self):
        """
        Gets list of filtered prices for each price tick on the primary grid
        """
        print("Deriving ranges of prices")
        for date in self.dates:
            print(date)
            if date == "2023-11-23":
                continue
            data = self.data.loc[date]
            if len(data) > 0:
                # reset index
                data = data.reset_index()
                # iterate over rows
                current_window_prices = []
                for row in range(data.shape[0]):
                    current_time = data.iloc[row, 1]
                    if row == 0:
                        window_center = current_time
                        for price in data.iloc[row, 2:]:
                            if math.isnan(price) == False:
                                current_window_prices += [price]
                    elif current_time >= window_center - self.window and current_time <= window_center + self.window:
                        for price in data.iloc[row, 2:]:
                            if math.isnan(price) == False:
                                current_window_prices += [price]
                    elif current_time > window_center + self.window:
                        # add last intervall to all prices
                        self.prices += [current_window_prices]
                        self.mean_prices += [np.mean(current_window_prices)]
                        # delete entries
                        current_window_prices = []
                        # update window center
                        window_center = window_center + self.frequency
                        # add filtered prices
                        if current_time >= window_center - self.window and current_time <= window_center + self.window:
                            for price in data.iloc[row, 2:]:
                                if math.isnan(price) == False:
                                    current_window_prices += [price]
                # add last intervall
                self.prices += [current_window_prices]
                print("Number of price intervals " + str(len(self.prices)))

    def get_list_of_prices_tick(self, raw=False):
        """
        Gets list of filtered tick prices for each price tick on the primary grid.
        :param raw: boolean whether the raw data should be returned or not
        """
        print("Deriving ranges of prices")
        number_empty_intervals = 0
        self.prices_for_anaysis = []
        for date in self.dates:
            day_prices = []
            print(date)
            if date == "2023-11-23":
                continue
            data = self.data.loc[date]
            if len(data) > 0:
                # reset index
                data = data.reset_index()
                # iterate over rows
                current_window_prices = []
                window_center = -1  # any starting value to ensure that the while loops works correctly
                for row in range(data.shape[0]):
                    current_time = data.iloc[row, 2]
                    assigned = False
                    iteration = 0
                    while (
                            window_center - self.window) < current_time or not assigned:  # to ensure that we get 23401 lists
                        iteration += 1
                        if iteration > 1000:
                            print("Something wrong too many iterations")
                            break
                        if row == 0:
                            window_center = 34200
                            if raw == True:
                                current_window_prices += [data.iloc[row, 3]]
                                break
                            else:
                                for price in data.iloc[row, 4:]:
                                    if math.isnan(price) == False:
                                        current_window_prices += [price]
                                break
                        elif current_time >= window_center - self.window and current_time <= window_center + self.window:
                            if raw == True:
                                current_window_prices += [data.iloc[row, 3]]
                                break
                            else:
                                for price in data.iloc[row, 4:]:
                                    if math.isnan(price) == False:
                                        current_window_prices += [price]
                                break
                        elif current_time > window_center + self.window:
                            # check if no ticks occured that it is filled with the last entry from the last available window
                            if len(current_window_prices) == 0:
                                number_empty_intervals += 1
                                current_window_prices += [self.prices[-1][-1]]
                            # add last intervall to all prices
                            self.prices += [current_window_prices]
                            day_prices += [current_window_prices]
                            self.mean_prices += [np.mean(current_window_prices)]
                            # delete entries
                            current_window_prices = []
                            # update window center
                            window_center = window_center + self.frequency
                            # add filtered prices
                            if current_time >= window_center - self.window and current_time <= window_center + self.window:
                                if raw == True:
                                    current_window_prices += [data.iloc[row, 3]]
                                else:
                                    for price in data.iloc[row, 4:]:
                                        if math.isnan(price) == False:
                                            current_window_prices += [price]
                                    assigned = True
                # add last intervall
                self.prices += [current_window_prices]
                day_prices += [current_window_prices]
                print("Number of price intervals " + str(len(self.prices)))
                self.prices_for_anaysis += [day_prices]
        print(f"Number of empty intervals {number_empty_intervals}")

    def get_date_range(self):
        """
        Get date range to collect filtered prices for
        """
        self.dates = pd.date_range(self.start_date, self.end_date, freq='d').strftime('%Y-%m-%d').tolist()



    def plot_filtered_opt(self, opt, frequency=1, daily_obs=23401, start_time='2023-12-12 15:58:46',
                          end_time='2023-12-12 15:59:01', price_intervals=None, save_path=None):
        """
        Plots the filtered and optimized prices within a given time range.
        :param opt: list of optimized prices
        :param frequency: frequency of the data
        :param daily_obs: number of daily observations
        :param start_time: start time of window to be plotted
        :param end_time: end time of window to be plotted
        :param price_intervals: sets of filtered prices used to generate moments of distribution
        :param save_path: path to save the plot
        """
        filtered_df = self.add_datetime_column(self.data, "Date", "Seconds")
        opt_datetimes = self.generate_optimized_datetimes(frequency, daily_obs)
        # Define the datetime range for plotting
        start_datetime = pd.Timestamp(start_time)
        end_datetime = pd.Timestamp(end_time)

        # Plot the filtered and optimized prices within the specified datetime range
        self.plot_prices(filtered_df, opt, opt_datetimes, start_datetime, end_datetime, frequency, price_intervals,
                         save_path)

    def plot_prices(self, filtered_df, optimized_prices, optimized_datetimes, start_datetime, end_datetime, frequency=1,
                    price_intervals=None, save_path=None):
        """
        Plots filtered and optimized prices within a given time range.

        :param filtered_df: dataframe containing filtered prices
        :param optimized_prices: list of optimized prices
        :param optimized_datetimes: list of datetime values for optimized prices
        :param start_datetime: the starting datetime for the window to be plotted
        :param end_datetime: the ending datetime for the window to be plotted
        :param frequency: frequency of the data
        :param price_intervals: list of price intervals
        :param save_path: path to save the plot
        """
        #collect the filtered and optimized prices by the datetime range
        filtered_df = filtered_df[
            (filtered_df['datetime'] >= start_datetime) & (filtered_df['datetime'] <= end_datetime)]
        # collect the optimized prices and corresponding datetimes
        optimized_filtered = [(price, dt) for price, dt in zip(optimized_prices, optimized_datetimes) if
                              start_datetime <= dt <= end_datetime]

        #Extract datetime and price values from the filtered optimized data
        optimized_times = [dt for price, dt in optimized_filtered]
        optimized_vals = [price for price, dt in optimized_filtered]

        #Plot the filtered prices
        plt.figure(figsize=(18, 9))

        price_cols = filtered_df.columns[2:-1]
        switch = 0
        for price_col in price_cols:
            if switch == 0:
                df_non_nan = filtered_df[~filtered_df[price_col].isna()]
                plt.plot(df_non_nan['datetime'], df_non_nan[price_col], 'bo', label='Filtered Prices', markersize=2)
                switch = 1
            else:
                df_non_nan = filtered_df[~filtered_df[price_col].isna()]
                plt.plot(df_non_nan['datetime'], df_non_nan[price_col], 'bo', markersize=2)

        #Plot the optimized prices
        plt.plot(optimized_times, optimized_vals, 'ro', label='Optimized Prices')

        #Add vertical lines for each full second in the plot
        for dt in optimized_times:
            plt.axvline(x=dt + timedelta(seconds=frequency / 2), color='gray', linestyle='--', alpha=0.5)

        #check if we want to plot elements of the distribution
        if price_intervals != None:
            filtered_intervals = [(interval, dt) for interval, dt in zip(price_intervals, optimized_datetimes) if
                                  start_datetime <= dt <= end_datetime]
            intervals = [price for price, dt in filtered_intervals]
            # get characteristics of distribution and plot it
            median = [np.median(interval) for interval in intervals]
            mean = [np.mean(interval) for interval in intervals]
            mean_max_min = [(np.max(interval) + np.min(interval)) / 2 for interval in intervals]
            plt.plot(optimized_times, median, marker="x", linestyle="None", label="Median", color="green")
            plt.plot(optimized_times, mean, marker="x", linestyle="None", label="Mean", color="purple")
            plt.plot(optimized_times, mean_max_min, marker="x", linestyle="None", label="Mean of Max and Min",
                     color="orange")

        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Log-Price', fontsize=14)
        plt.title('Filtered and Optimized Prices', fontsize=14)
        plt.legend(loc='best', fontsize=14)
        plt.xticks(rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(start_datetime, end_datetime)
        plt.tight_layout()

        # Show or save the plot
        if save_path != None:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()

    def add_datetime_column(self, df, date_col, seconds_col):
        """
        Adds a datetime column to the DataFrame based on the 'date' and 'seconds since midnight' columns.

        :param df: the DataFrame with 'date' and 'seconds since midnight' columns
        :param date_col: the name of the date column
        :param seconds_col: the name of the seconds since midnight column
        :return df: the DataFrame with an additional 'datetime' column
        """
        df['datetime'] = pd.to_datetime(df[date_col], format="%Y%m%d") + df[seconds_col].apply(
            lambda x: timedelta(seconds=34200 + x))
        return df

    def generate_optimized_datetimes(self, frequency, daily_obs, start_time='09:30:00'):
        """
        Generates a list of datetime values for optimized prices. For the default: each day starts at 9:30:00 and ends with 23401 price points, 1 second apart.
        :param start_time: start time for each day (e.g., '09:30:00')
        :param num_days: number of days to generate datetime column for
        :param num_seconds_per_day: number of seconds per day
        :return optimized_datetimes: list of datetime objects corresponding to the optimized primary grid
        """
        optimized_datetimes = []
        for day in pd.date_range(start=self.start_date, end=self.end_date).to_list():
            current_day = pd.Timestamp(day)
            base_time = pd.to_datetime(f"{current_day.date()} {start_time}")
            day_datetimes = [base_time + timedelta(seconds=i * frequency) for i in range(daily_obs)]
            optimized_datetimes.extend(day_datetimes)

        return optimized_datetimes






class Plot():
    """
    Class that implements the relevant plots.
    """
    def __init__(self, prices, frequency, lags, roh_data=False, len_days=None, path=None, save=False,
                 lm_test_path=None):
        """
        :param prices: list of prices to be plotted
        :param frequency: frequency of the data
        :param lags: number of lags to be considered in the ACF plot
        :param roh_data: boolean whether the data is ROH data or not
        :param len_days: list of lengths of the ROH data
        :param path: path to save the plot
        :param save: boolean whether to save the plot or not
        :param lm_test_path: path to save the LM test results
        """
        self.prices = prices
        self.frequency = frequency
        self.roh_data = roh_data
        if self.roh_data:
            self.daily_obs = len_days
            print(self.daily_obs)
        else:
            self.daily_obs = 23400 / frequency + 1
        self.returns = []
        self.lags = lags
        self.save_results = save
        self.path = path
        self.lm_test_path = lm_test_path

    def calculate_returns(self):
        """
        Functions that calculates the returns of prices
        """
        if self.roh_data:
            cumulative_list = []
            total = 0
            for length in self.daily_obs:
                total += length
                cumulative_list.append(total)
            day = 0
            for i, price in enumerate(self.prices):
                if i % cumulative_list[day] == 0:
                    if i == 0:
                        continue
                    else:
                        day += 1
                        print(day)
                else:
                    self.returns += [self.prices[i] - self.prices[i - 1]]
        else:
            for i, price in enumerate(self.prices):
                if i % self.daily_obs == 0:
                    continue
                else:
                    self.returns += [self.prices[i] - self.prices[i - 1]]

    def acf(self, name):
        """
        Function that generates ACF plots
        :param name: name of the plot
        """
        plot_acf(x=self.returns, zero=False, lags=self.lags, auto_ylims=True)
        plt.title("Autocorrelation of " + name, fontsize=12)
        plt.ylabel("Correlation", fontsize=12)
        plt.xlabel("Lag", fontsize=12)
        if self.save_results:
            plt.savefig(self.path + ".png", dpi=300)
        else:
            plt.show()

    def lm_test(self):
        """
        Function that generates LM test results
        """
        test = acorr_ljungbox(self.returns, lags=10)
        test.to_csv(self.lm_test_path + "lb.csv")
        lm = acorr_lm(self.returns, nlags=1)
        df = pd.DataFrame({'lm': [lm[0]], 'lmpval': [lm[1]]})
        path = self.lm_test_path + "lm_1.csv"
        df.to_csv(path)
        lm = acorr_lm(self.returns, nlags=10)
        df = pd.DataFrame({'lm': [lm[0]], 'lmpval': [lm[1]]})
        path = self.lm_test_path + "lm_10.csv"
        df.to_csv(path)
        print(test)

    def get_plot(self, name):
        self.calculate_returns()
        if self.lm_test_path != None:
            self.lm_test()
        self.acf(name)


    def calculate_realized_volatility_multi_days_multiple_schemes(self,
            path_sampled, path_filtered, path_optimized, schemes,
            start_date, end_date, daily_obs_optimized, other_weeks=False, tick=False, path_tick_filtered=None
    ):
        """
        Function that calculates the realized volatility for multiple schemes and multiple days.
        :param path_sampled: path to the sampled data
        :param path_filtered: path to the filtered data
        :param path_optimized: path to the optimized data
        :param schemes: list of schemes
        :param start_date: str of start date in the format YYYY-MM-DD
        :param end_date: str of end date in the format YYYY-MM-DD
        :param daily_obs_optimized: daily observations for the optimized data
        :param other_weeks: boolean whether the data is from other weeks (week 1 to 3) or not
        :param tick: boolean whether to include tick data or not
        :param path_tick_filtered: path to the tick filtered data
        """
        # Convert date strings to datetime objects
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

        plot_dict = {
            "sampled": {scheme: None for scheme in schemes},
            "filtered": {scheme: None for scheme in schemes},
            "optimized": None
        }

        # Load sampled data
        for scheme in schemes:
            df_sampled = pd.read_csv(path_sampled + "1_seconds/" + scheme + ".csv", header=None)
            df_sampled.columns = ['Date', 'Seconds', 'Price']
            df_sampled["Date"] = pd.to_datetime(df_sampled['Date'], format='%Y%m%d')
            df_sampled = df_sampled[(df_sampled['Date'] >= start_date) & (df_sampled['Date'] <= end_date)]
            df_sampled['Price'] = np.log(df_sampled['Price'])
            plot_dict["sampled"][scheme] = df_sampled

        # Load filtered data
        for scheme in schemes:
            if other_weeks == True:
                df_filtered = pd.read_csv(path_filtered + "1_seconds/moving_window_kalman_filter_" + scheme + "_new.csv")
            else:
                df_filtered = pd.read_csv(path_filtered + "1_seconds/moving_window_kalman_filter_" + scheme + ".csv")
            df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], format='%Y%m%d')
            df_filtered = df_filtered[(df_filtered['Date'] >= start_date) & (df_filtered['Date'] <= end_date)]
            plot_dict["filtered"][scheme] = df_filtered

        # Load optimized data
        df_optimized = pd.read_csv(path_optimized + "true_prices_1seconds.csv", header=0)
        optimized_prices = list(df_optimized.iloc[:, 0])

        if tick == True:
            df_optimized_tick = pd.read_csv(path_optimized + "tick_true_prices_1seconds.csv", header=0)
            optimized_prices_tick = list(df_optimized_tick.iloc[:, 0])
            df_optimized_tick_unfiltered = pd.read_csv(path_optimized + "tick_unfiltered_true_prices_1seconds.csv",
                                                       header=0)
            optimized_prices_tick_unfiltered = list(df_optimized_tick_unfiltered.iloc[:, 0])

            # Load Tick data here and compute RV
            tick_data = pd.read_csv(path_tick_filtered)
            tick_data['Date'] = pd.to_datetime(tick_data['Date'], format='%Y%m%d')
            tick_data = tick_data[(tick_data['Date'] >= start_date) & (tick_data['Date'] <= end_date)]
            tick_prices_observations = [group.iloc[:, 3].tolist() for _, group in tick_data.groupby("Date")]
            tick_prices_filtered = [group.iloc[:, 4].tolist() for _, group in tick_data.groupby("Date")]
            RV_tick_observations = []
            RV_tick_filtered = []
            for day in tick_prices_observations:
                # Compute log returns
                log_returns = np.diff(day)
                # Compute realized volatility as the square root of the sum of squared returns
                daily_volatility = np.sqrt(np.sum(log_returns ** 2))
                RV_tick_observations.append(daily_volatility)
            for day in tick_prices_filtered:
                # Compute log returns
                log_returns = np.diff(day)
                # Compute realized volatility as the square root of the sum of squared returns
                daily_volatility = np.sqrt(np.sum(log_returns ** 2))
                RV_tick_filtered.append(daily_volatility)

        # Helper function to compute realized volatility
        def compute_realized_volatility(prices, max_timestep=40, step=1):
            """
            Compute realized volatility for a given set of prices.
            :param prices: array of prices
            :param max_timestep: aggregation limit
            :param step: step size for aggregation
            :return: list of realized volatilities
            """
            volatilities = []
            for timestep in range(1, max_timestep + 1, step):
                squared_returns = []
                for shift in range(timestep):
                    returns = np.array(prices[shift::timestep][1:]) - np.array(prices[shift::timestep][:-1])
                    squared_returns.append(np.sum(returns ** 2))
                realized_volatilities = np.sqrt(squared_returns)
                volatilities.append(realized_volatilities)
            return volatilities

        # Process daily data for sampled, filtered, and optimized prices
        def process_daily_data(df, price_column, date_column):
            """
            Process daily data to compute realized volatility for each day.
            :param df: dataframe containing prices and dates
            :param price_column: column name for prices
            :param date_column: column name for dates
            :return: dictionary of daily volatilities
            """
            daily_volatilities = {}
            for day, group in df.groupby(df[date_column].dt.date):
                daily_prices = group[price_column].tolist()
                daily_volatilities[day] = compute_realized_volatility(daily_prices)
            return daily_volatilities

        # Process sampled and filtered data by date
        for scheme in plot_dict['sampled'].keys():
            plot_dict['sampled'][scheme] = process_daily_data(plot_dict['sampled'][scheme], 'Price', 'Date')
        for scheme in plot_dict['filtered'].keys():
            plot_dict['filtered'][scheme] = process_daily_data(plot_dict['filtered'][scheme],
                                                               plot_dict['filtered'][scheme].columns[-2], 'Date')

        # Process optimized data by splitting into daily segments
        optimized_vols_daily = {}
        day_counter = 0
        for i in range(0, len(optimized_prices), daily_obs_optimized):
            day_prices = optimized_prices[i:i + daily_obs_optimized]
            if len(day_prices) == daily_obs_optimized:
                day = start_date + timedelta(days=day_counter)
                optimized_vols_daily[day] = compute_realized_volatility(day_prices)
                day_counter += 1

        if tick:
            optimized_vols_daily_tick = {}
            day_counter = 0
            for i in range(0, len(optimized_prices_tick), daily_obs_optimized):
                day_prices = optimized_prices_tick[i:i + daily_obs_optimized]
                if len(day_prices) == daily_obs_optimized:
                    day = start_date + timedelta(days=day_counter)
                    optimized_vols_daily_tick[day] = compute_realized_volatility(day_prices)
                    day_counter += 1

            optimized_vols_daily_tick_unfiltered = {}
            day_counter = 0
            for i in range(0, len(optimized_prices_tick_unfiltered), daily_obs_optimized):
                day_prices = optimized_prices_tick_unfiltered[i:i + daily_obs_optimized]
                if len(day_prices) == daily_obs_optimized:
                    day = start_date + timedelta(days=day_counter)
                    optimized_vols_daily_tick_unfiltered[day] = compute_realized_volatility(day_prices)
                    day_counter += 1

        # Aggregate realized volatilities across days by timestep
        def aggregate_volatilities(vols_daily):
            """
            Aggregate and compute mean realized volatilities across days by timestep.
            :param vols_daily: dictionary of daily volatilities
            :return: dictionary of aggregated mean volatilities
            """
            timesteps = range(1, 41)
            aggregated_vols = {t: [] for t in timesteps}
            for daily_vols in vols_daily.values():
                for t, vols in zip(timesteps, daily_vols):
                    aggregated_vols[t].extend(vols)
            mean_vols = {t: np.mean(aggregated_vols[t]) for t in timesteps}
            return mean_vols

        for scheme in plot_dict['sampled'].keys():
            plot_dict['sampled'][scheme] = aggregate_volatilities(plot_dict['sampled'][scheme])
            plot_dict['filtered'][scheme] = aggregate_volatilities(plot_dict['filtered'][scheme])
        plot_dict['optimized'] = aggregate_volatilities(optimized_vols_daily)
        if tick:
            plot_dict['optimized_tick'] = aggregate_volatilities(optimized_vols_daily_tick)
            plot_dict['optimized_tick_unfiltered'] = aggregate_volatilities(optimized_vols_daily_tick_unfiltered)

        line_styles = {
            "optimized": "solid",
            "sampled": "dashed",
            "filtered": "dotted"
        }

        # Plot aggregated results
        timesteps = list(plot_dict['optimized'].keys())
        plt.figure(figsize=(12, 8))
        for scheme in plot_dict['sampled'].keys():
            plt.plot(timesteps, list(plot_dict['sampled'][scheme].values()), linestyle=line_styles["sampled"],
                     label='Sampled ' + scheme)
            plt.plot(timesteps, list(plot_dict['filtered'][scheme].values()), linestyle=line_styles["filtered"],
                     label='Filtered ' + scheme)
        plt.plot(timesteps, list(plot_dict['optimized'].values()), linestyle=line_styles["optimized"],
                 label='Optimized Prices')
        if tick:
            plt.plot(timesteps, list(plot_dict['optimized_tick'].values()), linestyle=line_styles["optimized"],
                     label='Optimized Tick Prices')
            plt.plot(1, np.mean(RV_tick_filtered), marker='x', label="Tick filtered")
            print(f"Mean RV tick {np.mean(RV_tick_observations)}")
        plt.xlabel("Timestep (seconds)")
        plt.ylabel("Mean Realized Volatility")
        plt.title("Mean Realized Volatility - Daily Aggregated")
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_acf_plots_per_week(self,path_sampled, path_filtered, path_optimized, path_tick_filtered, schemes, frequencies,
                               start_date, end_date, save_path=None, other_weeks=False):
        """
        Get all ACF plots per week: sampled,filtered, optimized and tick

        :param path_sampled: path to the sampled data
        :param path_filtered: path to the filtered data
        :param path_optimized: path to the optimized data
        :param path_tick_filtered: path to the tick filtered data
        :param schemes: list of schemes
        :param frequencies: list of frequencies
        :param start_date: str of start date in the format YYYY-MM-DD
        :param end_date: str of end date in the format YYYY-MM-DD
        :param save_path: path to save the plots
        :param other_weeks: boolean whether the data is from other weeks (week 1 to 3) or not
        """

        def acf_plots(prices, daily_obs, scheme, freq, key1, lags=10, save_path=None, tick=False):
            """
            Function to plot the ACF of the given prices.

            :param prices: array of prices
            :param daily_obs: daily observations
            :param scheme: name of scheme
            :param freq: frequency
            :param key1: key that identifies which type of price series
            :param lags: number of lags to plot
            :param save_path: path to save the plot
            :param tick: boolean whether to use tick data or not
            """
            if tick:
                returns = prices
            else:
                # calculate returns
                returns = []

                # Split prices into days
                num_days = len(prices) // daily_obs
                for i in range(num_days):
                    daily_prices = prices[i * daily_obs: (i + 1) * daily_obs]
                    returns_day = np.diff(daily_prices)
                    returns.extend(returns_day)

            # plot ACF
            if freq == 1:
                str_second = "second"
            else:
                str_second = "seconds"
            plot_acf(returns, zero=False, lags=lags, auto_ylims=True)
            if key1 == "tick":
                plt.title("ACF of tick data", fontsize=12)
            elif key1 == "tick_filtered":
                plt.title("ACF of filtered tick data", fontsize=12)
            elif key1 == "optimized_tick":
                plt.title("ACF of optimized tick data", fontsize=12)
            elif key1 != "optimized":
                plt.title("ACF of " + str(key1) + " " + str(scheme) + " " + str(freq) + " " + str_second, fontsize=12)
            else:
                plt.title("ACF of " + str(key1) + " " + str(freq) + " " + str_second, fontsize=12)
            plt.ylabel("Correlation", fontsize=12)
            plt.xlabel("Lag", fontsize=12)
            plt.tight_layout()
            if save_path != None:
                plt.savefig(save_path, dpi=300)
            else:
                plt.show()

        # parameters
        daily_obs = {1: 23401, 30: 781, 60: 391, 180: 131, 300: 79, 600: 40}

        start_date_copy = start_date
        end_date_copy = end_date
        # Convert date strings to datetime objects
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

        # read data
        # initiate container
        data = {"sampled": {freq: {} for freq in frequencies}, "filtered": {freq: {} for freq in frequencies}}
        # load/read sampled data
        for scheme in schemes:
            for freq in frequencies:
                df_sampled = pd.read_csv(path_sampled + str(freq) + "_seconds/" + scheme + ".csv", header=None)
                df_sampled.columns = ['Date', 'Seconds', 'Price']
                df_sampled["Date"] = pd.to_datetime(df_sampled['Date'], format='%Y%m%d')
                df_sampled = df_sampled[(df_sampled['Date'] >= start_date) & (df_sampled['Date'] <= end_date)]
                df_sampled['Price'] = np.log(df_sampled['Price'])
                data["sampled"][freq][scheme] = list(df_sampled['Price'])

        # Load filtered data
        for scheme in schemes:
            for freq in frequencies:
                if other_weeks and (freq == 1):
                    df_filtered = pd.read_csv(
                        path_filtered + str(freq) + "_seconds/moving_window_kalman_filter_" + scheme + "_new.csv")
                else:
                    df_filtered = pd.read_csv(
                        path_filtered + str(freq) + "_seconds/moving_window_kalman_filter_" + scheme + ".csv")
                df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], format='%Y%m%d')
                df_filtered = df_filtered[(df_filtered['Date'] >= start_date) & (df_filtered['Date'] <= end_date)]
                for column in range(df_filtered.shape[1] - 4):
                    if column == 0:
                        data["filtered"][freq][scheme] = [list(df_filtered.iloc[:, 4 + column])]
                    else:
                        data["filtered"][freq][scheme] += [list(df_filtered.iloc[:, 4 + column])]

        # Load optimized data
        df_optimized = pd.read_csv(path_optimized + "true_prices_1seconds.csv", header=0)
        optimized_prices = list(df_optimized.iloc[:, 0])
        data["optimized"] = optimized_prices

        # Load optimized data
        df_optimized_tick = pd.read_csv(path_optimized + "tick_true_prices_1seconds.csv", header=0)
        optimized_prices_tick = list(df_optimized_tick.iloc[:, 0])
        data["optimized_tick"] = optimized_prices_tick

        # Load Tick data here and compute RV
        tick_data = pd.read_csv(path_tick_filtered)
        tick_data['Date'] = pd.to_datetime(tick_data['Date'], format='%Y%m%d')
        tick_data = tick_data[(tick_data['Date'] >= start_date) & (tick_data['Date'] <= end_date)]
        tick_prices_observations = [group.iloc[:, 3].tolist() for _, group in tick_data.groupby("Date")]
        tick_prices_filtered = [group.iloc[:, 4].tolist() for _, group in tick_data.groupby("Date")]
        tick_returns = []
        tick_filtered_returns = []
        for day in tick_prices_observations:
            # Compute log returns
            log_returns = np.diff(day)
            tick_returns.extend(list(log_returns[:]))
        data["tick"] = tick_returns
        for day in tick_prices_filtered:
            # Compute log returns
            log_returns = np.diff(day)
            tick_filtered_returns.extend(list(log_returns[:]))
        data["tick_filtered"] = tick_filtered_returns

        # plot ACFs
        if save_path != None:
            for type in data.keys():
                if type == "optimized":
                    acf_plots(data["optimized"], 23401, None, 1, type, 10, save_path + "ACF_" + str(type) + "_1.png")
                elif type == "optimized_tick":
                    acf_plots(data["optimized_tick"], 23401, None, 1, type, 10,
                              save_path + "ACF_" + str(type) + "_1.png")
                elif type == "tick":
                    acf_plots(data["tick"], 23401, None, 1, type, 10, save_path + "ACF_" + str(type) + ".png",
                              tick=True)
                elif type == "tick_filtered":
                    acf_plots(data["tick_filtered"], 23401, None, 1, type, 10, save_path + "ACF_" + str(type) + ".png",
                              tick=True)
                else:
                    for freq in data[type].keys():
                        if type == "sampled":
                            for scheme in data["sampled"][freq].keys():
                                acf_plots(data["sampled"][freq][scheme], daily_obs[freq], scheme, freq, type, 10,
                                          save_path + "ACF_" + str(type) + "_" + str(freq) + "_" + str(scheme) + ".png")
                        if type == "filtered":
                            for scheme in data["filtered"][freq].keys():
                                acf_plots(data["filtered"][freq][scheme][0], daily_obs[freq], scheme, freq, type, 10,
                                          save_path + "ACF_" + str(type) + "_" + str(freq) + "_" + str(scheme) + ".png")

    def plot_interval_opt_filtered(self,path_filtered, path_optimized,frequency, start_date, end_date, daily_obs, start_time, end_time):
        """
        This function plots the optimized prices with the filtered prices. For a given date and a certain time window.
        :param frequency: frequency of the data
        :param start_date: start date to collect that data for
        :param end_date: end date to collect that data for
        :param daily_obs: number of daily observations
        :param start_time: time to start plotting
        :param end_time: time to end plotting
        """
        #get filtered data
        filtered_data = Collect_prices(path_filtered, start_date, end_date,
                                       frequency, frequency / 2, path="Plots/")
        #get optimized prices
        df_optimized = pd.read_csv(path_optimized, header=0)
        optimized_prices = list(df_optimized.iloc[:, 0])
        #plot filtered and optimized prices
        filtered_data.plot_filtered_opt(optimized_prices, frequency, daily_obs, start_time, end_time)

    def plot_interval_opt_filtered_with_moments_of_price_intervals(self,path_filtered,path_optimized,frequency, start_date, end_date, daily_obs,
                                                                   start_time, end_time, save_path=None):
        """
        This function plots the optimized prices with the filtered prices and moment of the data to get an idea what
        the optimized prices represent from the original data.
        :param path_filtered: path to the filtered data
        :param path_optimized: path to the optimized data
        :param frequency: frequency of the data
        :param start_date: start date to collect that data for
        :param end_date: end date to collect that data for
        :param daily_obs: number of daily observations
        :param start_time: time to start plotting
        :param end_time: time to end plotting
        :param save_path: path to save the plot
        """
        # get filtered data
        filtered_data = Collect_prices(path_filtered, start_date, end_date,
                                       frequency, frequency / 2, path="Plots/")
        filtered_data_extract = Collect_prices(path_filtered, start_date,
                                               end_date, frequency, frequency / 2, path="Plots/")
        price_ranges = filtered_data_extract.run_collect_prices()
        # get optimized prices
        df_optimized = pd.read_csv(path_optimized, header=0)
        optimized_prices = list(df_optimized.iloc[:, 0])
        # plot filtered and optimized prices
        filtered_data.plot_filtered_opt(optimized_prices, frequency, daily_obs, start_time, end_time, price_ranges,
                                        save_path)

    def plot_mean_max_min_acf(self,path_filtered,start_data,end_date,path_to_save):
        """
        This function plots the ACF of the mean max-min of the set of filtered prices.
        :param path_filtered: path to the filtered data
        :param start_data: start date to collect that data for
        :param end_date: end date to collect that data for
        :param path_to_save: path to save the plot
        """
        # get filtered data
        filtered_data_extract = Collect_prices(path_filtered, start_data,
                                               end_date, 1, 1/2, path="Plots/")
        price_ranges = filtered_data_extract.run_collect_prices()
        # compute mean max-min of the price ranges
        mean_max_min = [(np.max(interval) + np.min(interval)) / 2 for interval in price_ranges]
        # plot ACF
        acf = Plot(mean_max_min, 1, 10, save=False, path=path_to_save)
        acf.get_plot("Mean max-min 1 second")





