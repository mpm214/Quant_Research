import pandas as pd
from datetime import datetime
import numpy as np
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import LinearRegression
from pandas.tseries.offsets import DateOffset

# Example Usage: Assumes today's date since need forecasted date for example of using price prediction. 
fixed_today_date = datetime(2025, 2, 5).date()

# Load natural gas price data
data_path = "Quant_Research/Nat_Gas.csv"
nat_gas_prices = pd.read_csv(data_path)
nat_gas_prices['Dates'] = pd.to_datetime(nat_gas_prices['Dates'], format='%m/%d/%y')
nat_gas_prices.set_index('Dates', inplace=True)

# Decomposition using STL for forecasting
stl = STL(nat_gas_prices['Prices'], seasonal=13, robust=True)
result = stl.fit()
trend = result.trend
seasonal = result.seasonal

# Forecast future prices
time_index = np.arange(len(trend.dropna()))[:, np.newaxis]
lin_reg = LinearRegression()
lin_reg.fit(time_index, trend.dropna())
extended_time_index = np.arange(len(trend.dropna()) + 12)[:, np.newaxis]
predicted_trend = lin_reg.predict(extended_time_index)

# Future forecasted prices (12 months ahead)
future_dates = pd.date_range(start=nat_gas_prices.index[-1] + DateOffset(months=1), periods=12, freq='ME')
seasonal_repeated = np.tile(seasonal.values[:12], 1)[:12]
total_forecast = predicted_trend[-12:] + seasonal_repeated
forecasted_prices = pd.Series(total_forecast, index=future_dates)

# Function to get price for a given date
def get_price_for_date(date):
    date = pd.to_datetime(date)
    if date in nat_gas_prices.index:
        return nat_gas_prices.loc[date]['Prices']
    interpolated_price = np.interp(date.toordinal(), nat_gas_prices.index.map(datetime.toordinal), nat_gas_prices['Prices'])
    return interpolated_price

# Log a new purchase of natural gas
def log_purchase(purchase_log, purchase_date, amount_mmbtu, storage_capacity=15_000_000):
    purchase_date = datetime.strptime(purchase_date, '%Y-%m-%d').date()
    price_per_mmbtu = get_price_for_date(purchase_date)
    
    total_remaining_units = purchase_log['Remaining_Units'].sum() if not purchase_log.empty else 0
    storage_availability = storage_capacity - total_remaining_units
    
    new_purchase = {
        'Purchase_Date': purchase_date,
        'Price_per_MMBtu': price_per_mmbtu,
        'Amount_MMBtu': amount_mmbtu,
        'Purchase_Cost': price_per_mmbtu * amount_mmbtu,
        'Sold': '',
        'Remaining_Units': amount_mmbtu,
        'Sale_IDs': '',
        'Request_IDs': '',
        'Storage_Availability': storage_availability
    }
    return purchase_log._append(new_purchase, ignore_index=True)

# Function to recalculate storage availability after changes in remaining units (sales or withdrawals)
def update_storage_availability(purchase_log_df, storage_capacity=15_000_000):
    """Recalculates the storage availability based on updated remaining units."""
    for idx in range(len(purchase_log_df)):
        total_remaining_units = purchase_log_df['Remaining_Units'].iloc[:idx+1].sum()
        purchase_log_df.at[idx, 'Storage_Availability'] = storage_capacity - total_remaining_units
    return purchase_log_df

# Initialize purchase log
purchase_log_df = pd.DataFrame(columns=['Purchase_Date', 'Price_per_MMBtu', 'Amount_MMBtu', 'Purchase_Cost', 'Sold', 'Remaining_Units', 'Sale_IDs', 'Request_IDs', 'Storage_Availability'])

# Example Usage for logging purchases
purchase_log_df = log_purchase(purchase_log_df, '2024-09-04', 1_500_000)
purchase_log_df = log_purchase(purchase_log_df, '2024-09-10', 2_000_000)
purchase_log_df = log_purchase(purchase_log_df, '2024-09-15', 3_000_000)
purchase_log_df = log_purchase(purchase_log_df, '2024-09-20', 1_000_000)
purchase_log_df = log_purchase(purchase_log_df, '2024-09-25', 1_800_000)

# Function to get sale price for a sale date
def get_price_for_sale(sale_date):
    if sale_date in forecasted_prices.index:
        return forecasted_prices.loc[sale_date]
    interpolated_price = np.interp(sale_date.toordinal(), forecasted_prices.index.map(lambda x: x.toordinal()), forecasted_prices.values)
    return interpolated_price

# Initialize the sales log
sales_log_df = pd.DataFrame(columns=['Sale_ID', 'Sale_Date', 'Price_per_MMBtu', 'Amount_MMBtu', 'Sale_Proceeds', 'Cost_of_Sale'])

# Example Usage for logging sales
sale_dates = [pd.Timestamp('2025-01-30'), pd.Timestamp('2025-02-01')]
amounts_mmbtu = [5_000_000, 1_300_000]

for idx, (sale_date, amount_mmbtu) in enumerate(zip(sale_dates, amounts_mmbtu)):
    price_per_mmbtu = get_price_for_sale(sale_date)
    sale_proceeds = price_per_mmbtu * amount_mmbtu
    
    new_sale = {
        'Sale_ID': f'SALE_{idx+1}',
        'Sale_Date': sale_date,
        'Price_per_MMBtu': price_per_mmbtu,
        'Amount_MMBtu': amount_mmbtu,
        'Sale_Proceeds': sale_proceeds,
        'Cost_of_Sale': 0  # To be calculated later
    }
    
    sales_log_df = sales_log_df._append(new_sale, ignore_index=True)

# Function to update the purchase log with sales and calculate cost of sale
def update_purchase_log_with_sales(purchase_log_df, sales_log_df):
    for _, sale in sales_log_df.iterrows():
        sale_id = sale['Sale_ID']
        sale_amount = sale['Amount_MMBtu']
        total_cost_of_sale = 0
        
        # Iterate over purchase log entries (FIFO)
        for idx, purchase in purchase_log_df.iterrows():
            if purchase['Sold'] != 'Yes - FULL':
                remaining_units = purchase_log_df.at[idx, 'Remaining_Units']
                
                if sale_amount <= remaining_units:
                    pro_rata_cost = (sale_amount / purchase['Amount_MMBtu']) * purchase['Purchase_Cost']
                    total_cost_of_sale += pro_rata_cost
                    
                    purchase_log_df.at[idx, 'Remaining_Units'] -= sale_amount
                    purchase_log_df.at[idx, 'Sale_IDs'] += sale_id + ' '
                    sale_amount = 0
                    
                    if purchase_log_df.at[idx, 'Remaining_Units'] == 0:
                        purchase_log_df.at[idx, 'Sold'] = 'Yes - FULL'
                    else:
                        purchase_log_df.at[idx, 'Sold'] = 'Yes - PARTIALLY'
                    
                    break
                else:
                    pro_rata_cost = (remaining_units / purchase['Amount_MMBtu']) * purchase['Purchase_Cost']
                    total_cost_of_sale += pro_rata_cost
                    
                    purchase_log_df.at[idx, 'Remaining_Units'] = 0
                    purchase_log_df.at[idx, 'Sold'] = 'Yes - FULL'
                    sale_amount -= remaining_units
                    purchase_log_df.at[idx, 'Sale_IDs'] += sale_id + ' '
        
        sales_log_df.loc[sales_log_df['Sale_ID'] == sale_id, 'Cost_of_Sale'] = total_cost_of_sale
    
    purchase_log_df = update_storage_availability(purchase_log_df)
    return purchase_log_df, sales_log_df

# Update purchase log and sales log with sales
updated_purchase_log, updated_sales_log = update_purchase_log_with_sales(purchase_log_df, sales_log_df)

# Function to handle withdrawals and calculate profits
def check_valuation(purchase_log_df, units=None, withdrawals=None, storage_cost_per_month=100_000, transport_cost_per_withdrawal=50_000, injection_cost_per_purchase=10_000, throughput=1_000_000):
    if 'Request_IDs' not in purchase_log_df.columns:
        purchase_log_df['Request_IDs'] = ''
    
    remaining_log = purchase_log_df[purchase_log_df['Remaining_Units'] > 0]
    total_remaining_units = remaining_log['Remaining_Units'].sum()
    total_remaining_cost = (remaining_log['Remaining_Units'] * remaining_log['Price_per_MMBtu']).sum()
    
    avg_cost_all_unsold = total_remaining_cost / total_remaining_units if total_remaining_units > 0 else 0
    
    valuation_summary = {
        'Total_Remaining_Units': total_remaining_units,
        'Average_Cost_All_Unsold': avg_cost_all_unsold
    }
    
    withdrawal_log = []

    if withdrawals is None:
        withdrawals = [(fixed_today_date, units)]
    
    request_id_counter = 1
    for withdrawal_date, withdrawal_units in withdrawals:
        remaining_units_needed = withdrawal_units
        total_cost = 0
        total_storage_months_weighted = 0
        units_counted = 0
        
        purchase_transport_costs = []
        injection_costs = []

        # FIFO logic for withdrawals
        for idx, row in purchase_log_df.iterrows():
            if remaining_units_needed == 0:
                break
            
            available_units = purchase_log_df.at[idx, 'Remaining_Units']
            unit_cost = purchase_log_df.at[idx, 'Price_per_MMBtu']
            purchase_date = purchase_log_df.at[idx, 'Purchase_Date']
            request_id = f'REQUEST_{request_id_counter}'
            
            if available_units <= 0:
                continue

            storage_duration = (withdrawal_date - purchase_date).days / 30  # 30 / 360 Basis
            throughput_factor = purchase_log_df.at[idx, 'Amount_MMBtu'] / throughput
            
            if remaining_units_needed <= available_units:
                total_cost += remaining_units_needed * unit_cost
                total_storage_months_weighted += remaining_units_needed * storage_duration
                units_counted += remaining_units_needed
                
                purchase_transport_cost = (remaining_units_needed / purchase_log_df.at[idx, 'Amount_MMBtu']) * transport_cost_per_withdrawal
                purchase_transport_costs.append(purchase_transport_cost)

                injection_cost = (remaining_units_needed / purchase_log_df.at[idx, 'Amount_MMBtu']) * injection_cost_per_purchase * throughput_factor
                injection_costs.append(injection_cost)
                
                purchase_log_df.at[idx, 'Remaining_Units'] -= remaining_units_needed
                purchase_log_df.at[idx, 'Request_IDs'] = (purchase_log_df.at[idx, 'Request_IDs'] + f' {request_id}').strip()
                
                if purchase_log_df.at[idx, 'Remaining_Units'] == 0:
                    purchase_log_df.at[idx, 'Sold'] = f'Yes - FULL (Request)'
                else:
                    purchase_log_df.at[idx, 'Sold'] = f'Yes - PARTIALLY (Request)'
                
                remaining_units_needed = 0
            else:
                total_cost += available_units * unit_cost
                total_storage_months_weighted += available_units * storage_duration
                units_counted += available_units
                
                purchase_transport_cost = (available_units / purchase_log_df.at[idx, 'Amount_MMBtu']) * transport_cost_per_withdrawal
                purchase_transport_costs.append(purchase_transport_cost)

                injection_cost = (available_units / purchase_log_df.at[idx, 'Amount_MMBtu']) * injection_cost_per_purchase * throughput_factor
                injection_costs.append(injection_cost)
                
                purchase_log_df.at[idx, 'Remaining_Units'] = 0
                purchase_log_df.at[idx, 'Sold'] = f'Yes - FULL (Request)'
                purchase_log_df.at[idx, 'Request_IDs'] = (purchase_log_df.at[idx, 'Request_IDs'] + f' {request_id}').strip()
                
                remaining_units_needed -= available_units
        
        avg_cost_requested_units = total_cost / units_counted
        weighted_average_storage_months = total_storage_months_weighted / units_counted
        total_storage_cost = round(weighted_average_storage_months * storage_cost_per_month)
        withdrawal_transport_cost = transport_cost_per_withdrawal
        withdrawal_cost = 10_000 * (withdrawal_units / throughput)
        total_cost_per_withdrawal = (withdrawal_units * avg_cost_requested_units) + total_storage_cost + sum(purchase_transport_costs) + withdrawal_transport_cost + sum(injection_costs) + withdrawal_cost
        price_per_mmbtu = get_price_for_sale(withdrawal_date)
        sales_proceed = price_per_mmbtu * withdrawal_units
        net_profit = sales_proceed - total_cost_per_withdrawal
        
        withdrawal_log.append({
            'Withdrawal Date': withdrawal_date,
            'Requested Units': withdrawal_units,
            'Average_Cost': avg_cost_requested_units,
            'Total Storage Costs': total_storage_cost,
            'Purchase_Transport_Cost': purchase_transport_costs,
            'Withdrawal_Transport_Cost': withdrawal_transport_cost,
            'Injection_Cost': injection_costs,
            'Withdrawal_Cost': withdrawal_cost,
            'Total_Cost': total_cost_per_withdrawal,
            'Price_per_MMBtu': price_per_mmbtu,
            'Sales_Proceeds': sales_proceed,
            'Net_Profit': net_profit
        })
        
        request_id_counter += 1
    
    purchase_log_df = update_storage_availability(purchase_log_df)
    withdrawal_log_df = pd.DataFrame(withdrawal_log)
    
    total_net_profit = withdrawal_log_df['Net_Profit'].sum()
    
    return valuation_summary, withdrawal_log_df, purchase_log_df, total_net_profit

# Example usage with withdrawals
withdrawals = [
    (datetime(2025, 2, 5).date(), 1_000_000),
    (datetime(2025, 2, 12).date(), 1_500_000)
]

valuation_summary, withdrawal_log, updated_purchase_log, total_net_profit = check_valuation(updated_purchase_log, withdrawals=withdrawals)

# Print
print("Sales Log with Cost of Sale:")
print(updated_sales_log)

print("Valuation for all unsold units as of Initial Request Date:")
print(valuation_summary)

print("Withdrawal Log:")
print(withdrawal_log)

print("Total Net Profit:")
print(total_net_profit)

print("Purchase Log:")
print(updated_purchase_log)
