import time
from datetime import timedelta

import pandas as pd
from haversine import haversine

from shift import Shift



class Simulation:
    '''
    Class to model a simulation of the shift of a determined period.
    '''
    def __init__(self, depot, config, window_size=6, max_size=175, filter_function=None, filter_kwargs={}):
        self.depot = depot
        self.cluster_class = config['cluster_class']
        self.cluster_kwargs = config['cluster_kwargs']
        self.graph_class = config['graph_class']
        self.graph_kwargs = config['graph_kwargs']
        self.filter_function = filter_function
        self.window_size = window_size
        self.max_size = max_size
        self.filter_kwargs = filter_kwargs
        self.distances = []
        self.routes = []
        self.timetables = []

    def get_score(self):
        return sum(self.distances)

    def get_distances(self):
        return self.distances

    def get_results(self):
        return self.routes

    def get_timetables(self, speed=30, emp_time=60):
        return self.timetables

    def to_csv(self, file):
        with open(file, 'w') as f:
            csv_output = []
            for w in range(len(self.routes)):
                for v in range(len(self.routes[w])):
                    for o in range(len(self.routes[w][v])):
                        if(self.timetables):
                            csv_output.append(
                                f'{self.routes[w][v][o][0]},{self.routes[w][v][o][1]},{o+1},{v+1},{w+1}, {self.timetables[w][v][o]}\n')
                        else:
                            csv_output.append(
                                f'{self.routes[w][v][o][0]},{self.routes[w][v][o][1]},{o+1},{v+1},{w+1}\n')
            if(self.timetables):
                f.write('lat,long,order,vehicle,window,timetable\n')
            else:
                f.write('lat,long,order,vehicle,window\n')
            f.writelines(csv_output)


    def compute_simulation(self, data, start_date=None, end_date=None, speed=None, emp_time=None, debug=False):
        '''
        This method computes a simulation.
        It returns the routes of every shift.
        The simulation keep track of the necessity to empty every bin once a day, at least. Every not emptied bin is added to the last shift of the day.
        '''
        i = 0
        start_date = data.index[0] if start_date is None else pd.to_datetime(
            start_date)
        end_date = data.index[-1] if end_date is None else pd.to_datetime(
            end_date)

        current_start_date = start_date
        current_end_date = start_date + \
            timedelta(hours=self.window_size)-timedelta(seconds=1)

        # bins serials of all the bins
        bin_serials = data.bin_serial.unique()
        all_bins = data.drop_duplicates(subset='bin_serial')
        all_bins.loc[:, 'bin_level'] = 4

        # not emptied bins initialization
        not_full = all_bins.copy()

        while(True):
            if(debug):
                print(str(current_start_date) + " - " + str(current_end_date))
            current_window = data[current_start_date: current_end_date]

            current_start_date += timedelta(hours=self.window_size)
            current_end_date += timedelta(hours=self.window_size)
            # last window of the day?
            last_window = True if(current_end_date.day != (
                current_start_date-timedelta(seconds=1)).day) else False

            # if last shift, not emptied bins need to be added
            current_window = pd.concat(
                [current_window, not_full]) if last_window else current_window
            # bins to be emptied
            current_window = current_window if self.filter_function is None else self.filter_function(
                current_window, **self.filter_kwargs)
            # update the not emptied bins list
            not_full = all_bins.copy() if last_window else not_full[~not_full.bin_serial.isin(
                current_window['bin_serial'].unique())]

            if(current_window.shape[0] == 0):
                continue
            current_shift = Shift(current_window, self.depot, self.cluster_class,
                                  self.graph_class, start_date=current_start_date)

            # clustering
            start_time = time.time()
            clusters = current_shift.clusterize(
                columns=['latitude', 'longitude'], max_size=self.max_size, kwargs=self.cluster_kwargs)
            current_shift.build_graphs(
                columns=['latitude', 'longitude'], kwargs=self.graph_kwargs)

            # routing computing
            results = current_shift.get_routes()
            #results = [[self.depot, *result[1:-1], self.depot] for result in results]
            self.routes.append(results)
            self.distances.append(current_shift.get_distance())
            if(speed is not None or emp_time is not None):
                self.timetables.append(
                    current_shift.get_timetables(haversine, speed, emp_time))
            if(debug):
                print(
                    f"Distanza totale turno: {str(self.distances[-1])} km. Veicoli usati: {str(len(results))}.")
            if(debug):
                print(f"Tempo richiesto: {time.time() - start_time}s.")
            del current_shift

            if(current_end_date >= end_date):  # Stop condition
                break

        return self.routes

    def print_simulation(self):
        print('# Simulation #')
        print(f'Shifts executed: {str(len(self.routes))}.')
        print(f'Total distance: {str(self.score)} km.')
        total_bins = sum([len(self.routes[w][c]) for w in range(len(self.routes))
                          for c in range(len(self.routes[w]))])
        print(f'Bins emptied: {str(total_bins)}.')
        total_vehs = sum([len(self.routes[w])
                         for w in range(len(self.routes))])
        print(f'Vehicles involved: {str(total_vehs)}.')
