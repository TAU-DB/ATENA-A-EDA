"""
This file contains all data with regard to columns in the wide12 flight delays schema (a big flight schema with 20% of
the columns duplicated)
"""

base_KEYS = ['flight_id', 'airline', 'origin_airport', 'destination_airport', 'flight_number',
             'delay_reason', 'departure_delay',
             'scheduled_trip_time',
             'scheduled_departure', 'scheduled_arrival', 'day_of_week', 'day_of_year',
             'taxi_out', 'taxi_in', 'wheels_off',
             'wheels_on', 'year', 'month', 'elapsed_time', 'air_time', 'distance',
             'air_system_delay', 'security_delay', 'airline_delay',
             'late_aircraft_delay', 'weather_delay'
             ]
_extra_keys = ['departure_delay_2', 'year_2', 'taxi_in_2', 'distance_2',
       'origin_airport_2']
_extra_keys_STR = ['Departure Delay 2', 'Year 2', 'Arrival Taxi 2', 'Distance 2', 'Origin Airport 2'

]

KEYS = base_KEYS[::] + _extra_keys[::]

bsae_KEYS_ANALYST_STR = ['Flight ID', 'Airline', 'Origin Airport', 'Destination Airport', 'Flight Number',
                         'Delay Reason', 'Departure Delay',
                         'Scheduled Trip Time (Planned time amount needed for the flight)',
                         'Scheduled Departure', 'Scheduled Arrival', 'Day of Week', 'Day of Year',
                         'Departure Taxi', 'Arrival Taxi', 'Wheels Departure Time', 'Wheels Arrival Time',
                         'Year', 'Month', 'Elapsed Time',
                         'Air Time', 'Distance', 'Air-System Delay', 'Security Delay', 'Airline Delay',
                         'Late Aircraft Delay', 'Weather Delay'
                         ]

KEYS_ANALYST_STR = bsae_KEYS_ANALYST_STR[::] + _extra_keys_STR[::]

assert len(KEYS) == len(KEYS_ANALYST_STR)
FILTER_COLS = KEYS  # Note: changing this from KEYS require to change other occurrences of KEYS in the codebase
GROUP_COLS = KEYS  # Note: changing this from KEYS require to change other occurrences of KEYS in the codebase

NUMERIC_KEYS = {}
AGG_KEYS = ['flight_id', 'flight_id_2']
AGG_KEYS_ANALYST_STR = ['Flight ID', 'Flight ID 2']
FILTER_LIST = [
    'AS', '29', '5', 'SJU', 'WEATHER', 'SMF', 'SNA', 'PHL', '12', 'MQ', '469', '16', 'MID_DELAY', '28', 'CLT', '188',
    'LGA', '18', 'CLE', 'AFTERNOON', 'PHX', 'VX', 'ON_TIME', 'EVENING', 'DAL', 'BOS', '4', 'DFW', '26', 'HNL', 'PBI',
    '11', 'PIT', 'MID_FLIGHT', '316', 'LONG_FLIGHT', '30', 'RSW', 'OAK', 'LATE_AIRCRAFT', 'TPA', 'DL', 'SJC', '1',
    'EWR', '404', 'FLL', '27', 'SAN', '19', '341', 'AIR_SYSTEM', 'MIA', 'NK', 'LAX', '61', 'SHORT_FLIGHT', 'MSY', 'WN',
    'DTW', 'SECURITY', 'PDX', 'IAD', 'None', '711', '15', 'IND', 'LAS', 'UA', 'MCO', '13', 'F9', 'MORNING', 'MKE',
    'BNA', 'MDW', 'NIGHT', '432', '23', '345', 'ATL', 'US', 'MSP', '10', 'BWI', 'EV', 'JFK', 'SMALL_DELAY', '20',
    '8', 'AIRLINE', 'SAT', 'AA', 'STL', 'ORD', 'HA', 'RDU', 'LARGE_DELAY', '9', '3', 'SLC', '24', 'SFO', 'B6',
    'DCA', 'HOU', 'OO', '21', '25', '7', 'IAH', '17', '6', 'DEN', 'AUS', '14', '745', 'MCI', '2', '22', 'SEA']

FILTER_BY_FIELD_DICT = {key: set() for key in KEYS}

for departure_delay_field in ['departure_delay', 'departure_delay_2']:
    FILTER_BY_FIELD_DICT[departure_delay_field] = {
        'ON_TIME',
        'LARGE_DELAY'
    }

DONT_FILTER_FIELDS = {'flight_id', 'flight_id_2'}
