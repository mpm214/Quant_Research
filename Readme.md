# Quantitative Researcher Course

## TASK 1

### Background Information
You are a quantitative researcher working with a commodity trading desk. Alex, a VP on the desk, wants to start trading natural gas storage contracts. However, the available market data must be of higher quality to enable the instrument to be priced accurately. They have sent you an email asking you to help extrapolate the data available from external feeds to provide more granularity, considering seasonal trends in the price as it relates to months in the year. To price the contract, we will need historical data and an estimate of the future gas price at any date.

Commodity storage contracts represent deals between warehouse (storage) owners and participants in the supply chain (refineries, transporters, distributors, etc.). The deal is typically an agreement to store an agreed quantity of any physical commodity (oil, natural gas, agriculture) in a warehouse for a specified amount of time. The key terms of such contracts (e.g., periodic fees for storage, limits on withdrawals/injections of a commodity) are agreed upon inception of the contract between the warehouse owner and the client. The injection date is when the commodity is purchased and stored, and the withdrawal date is when the commodity is withdrawn from storage and sold.

A client could be anyone who would fall within the commodities supply chain, such as producers, refiners, transporters, and distributors. This group would also include firms (commodities trading, hedge funds, etc.) whose primary aim is to take advantage of seasonal or intra-day price differentials in physical commodities. For example, if a firm is looking to buy physical natural gas during summer and sell it in winter, it would take advantage of the seasonal price differential mentioned above. The firm would need to leverage the services of an underground storage facility to store the purchased inventory to realize any profits from this strategy.

Reference Material: [Understanding Commodity Storage](https://www.cmegroup.com/education/courses/introduction-to-energy/introduction-to-crude-oil/understanding-commodity-storage.html)

### Understanding Commodity Storage
Storage facilities play a crucial role in the commodities supply, transportation, and consumption chain. Storage is a means of collecting products before distribution into downstream operations in the midstream sector. Storage is also used by downstream operators as an additional source of supply in the case of supply disruption.

#### Crude Oil Storage
Crude oil is generally stored in tanks with varying designs depending on the usage. Storage tanks come in all sizes and shapes. Each tank is designed to handle pressure conditions of the liquids, prevent leakage and corrosion, and manage ventilation and fumes. The most common tank is the vertical, cylindrical storage tank. It ranges in size from 10 to 400 feet in diameter. The storage capacity of these tanks also varies widely, from a few hundred barrels to several thousand.

Supertankers, like very large crude carriers (VLCC) and Ultra Large Crude Carriers (ULCC), are occasionally used as storage facilities. They are usually above 250,000 deadweight tonnage in size and can carry and store more than 2 million barrels.

In the early 2000s, United States crude oil storage maintained stable levels. A brief storage surge due to low prices ended in 2015, since then, crude oil storage levels have increased steadily with small fluctuations. The storage capacity utilization rate has been around 56% to 66% level during the past five years.

#### Petrochemical Storage
Compared to crude oil, petrochemical storage facilities are more diverse and complex. Take propane as an example. For industrial use, propane is usually stored in a cylinder tank. Whereas, for retail use, the storage tanks are called bulk plants, with typically 18,000 to 30,000 gallons capacity each.

Over the past five years, the entire petrochemicals sector has maintained stable storage levels after a slight decline in February 2013.

#### Natural Gas Storage
Natural gas typically uses underground storage facilities including salt caverns, mines, aquifers, depleted reservoirs, and hard-rock caverns. Natural gas storage plays an important role in meeting the seasonal demand.

Most natural gas is stored in depleted reservoirs. The reservoir’s infrastructure, like a well’s, gathering system, and pipeline connections can be reused. This is more cost-efficient than an aquifer which consists of water-bearing sedimentary rock.

The volume of natural gas used as the permanent inventory in storage is referred to as base gas. Working gas refers to the excess volume that can be extracted for operations, such as power burn and serving retail customers. Over the past few years, working gas and base gas have been increasing steadily after a long period of stability.

#### Electricity Storage
Not all commodities can be economically stored. For example, it is still prohibitively expensive to store electricity in an industrial-level battery plant. Thus, the volatility of electricity prices is generally higher than other commodities, since it is prone to supply and demand disruption risk.

Major price swings, as much as 600% in one day, can occur in electricity. This occurred on May 19, 2017, due to an unexpected heat wave.

#### Conclusion
Commodity storage plays an important role in balancing supply and demand. When supply is disrupted by unexpected events, a sufficient storage level will reduce the financial and physical impacts felt by downstream operators while allowing them to continue their operations. They also maintain the option to physically store commodities in a low-price environment and wait for the price to rebound to generate greater profits.

### Your Task
After asking around for the source of the existing data, you learn that the current process is to take a monthly snapshot of prices from a market data provider, which represents the market price of natural gas delivered at the end of each calendar month. This data is available for roughly the next 18 months and is combined with historical prices in a time series database. After gaining access, you are able to download the data in a CSV file.

You should use this monthly snapshot to produce a varying picture of the existing price data, as well as an extrapolation for an extra year, in case the client needs an indicative price for a longer-term storage contract.

- Download the monthly natural gas price data.
- Each point in the data set corresponds to the purchase price of natural gas at the end of a month, from 31st October 2020 to 30th September 2024.
- Analyze the data to estimate the purchase price of gas at any date in the past and extrapolate it for one year into the future.
- Your code should take a date as input and return a price estimate.

Try to visualize the data to find patterns and consider what factors might cause the price of natural gas to vary. This can include looking at months of the year for seasonal trends that affect the prices, but market holidays, weekends, and bank holidays need not be accounted for. Submit your completed code below.

## TASK 2: Price a commodity storage contract

**What you'll learn**
How to write a function that takes particular inputs and gives back the value of a contract

**What you'll do**
Create a prototype pricing model that can go through further validation and testing before being put into production


### Background Information

Great work! The desk now has the price data they need. The final ingredient before they can begin trading with the client is the pricing model. Alex tells you the client wants to start trading as soon as possible. They believe the winter will be colder than expected, so they want to buy gas now to store and sell in winter in order to take advantage of the resulting increase in gas prices. They ask you to write a script that they can use to price the contract. Once the desk are happy, you will work with engineering, risk, and model validation to incorporate this model into production code.

The concept is simple: any trade agreement is as valuable as the price you can sell minus the price at which you are able to buy. Any cost incurred as part of executing this agreement is also deducted from the overall value. So, for example, if I can purchase a million MMBtu of natural gas in summer at $2/MMBtu, store this for four months, and ensure that I can sell the same quantity at $3/MMBtu without incurring any additional costs, the value of this contract would be ($3-$2) *1e6 = $1million. If there are costs involved, such as having to pay the storage facility owner a fixed fee of $100K a month, then the 'value' of the contract, from my perspective, would drop by the overall rental amount to $600K. Another cost could be the injection/withdrawal cost, like having to pay the storage facility owner $10K per 1 million MMBtu for injection/withdrawal, then the price will further go down by $10K to $590K. Additionally, if I am supposed to foot a bill of $50K each time for transporting the gas to and from the facility, the cost of this contract would fall by another $100K. Think of the valuation as a fair estimate at which both the trading desk and the client would be happy to enter into the contract. 

### Task

You need to create a prototype pricing model that can go through further validation and testing before being put into production. Eventually, this model may be the basis for fully automated quoting to clients, but for now, the desk will use it with manual oversight to explore options with the client. 

You should write a function that is able to use the data you created previously to price the contract. The client may want to choose multiple dates to inject and withdraw a set amount of gas, so your approach should generalize the explanation from before. Consider all the cash flows involved in the product.

The input parameters that should be taken into account for pricing are:

- Injection dates. 
- Withdrawal dates.
- The prices at which the commodity can be purchased/sold on those dates.
- The rate at which the gas can be injected/withdrawn.
- The maximum volume that can be stored.
- Storage costs.
- Write a function that takes these inputs and gives back the value of the contract. You can assume there is no transport delay and that interest rates are zero. Market holidays, weekends, and bank holidays need not be accounted for. Test your code by selecting a few sample inputs.

## TASK 3: Credit Risk Analysis

**What you'll learn**
How to choose appropriate independent variables from a data set that will accurately predict the outcome of a chosen dependent variable 
The importance of using available data to predict customer trends and actions

**What you'll do**
Build a model using Python that will estimate the probability of default for a borrower

### Background Information

You have now moved to a new team assisting the retail banking arm, which has been experiencing higher-than-expected default rates on personal loans. Loans are an important source of revenue for banks, but they are also associated with the risk that borrowers may default on their loans. A default occurs when a borrower stops making the required payments on a debt.

The risk team has begun to look at the existing book of loans to see if more defaults should be expected in the future and, if so, what the expected loss will be. They have collected data on customers and now want to build a predictive model that can estimate the probability of default based on customer characteristics. A better estimate of the number of customers defaulting on their loan obligations will allow us to set aside sufficient capital to absorb that loss. They have decided to work with you in the QR team to help predict the possible losses due to the loans that would potentially default in the next year.

Charlie, an associate in the risk team, who has been introducing you to the business area, sends you a small sample of their loan book and asks if you can try building a prototype predictive model, which she can then test and incorporate into their loss allowances.

### Task

The risk manager has collected data on the loan borrowers. The data is in tabular format, with each row providing details of the borrower, including their income, total loans outstanding, and a few other metrics. There is also a column indicating if the borrower has previously defaulted on a loan. You must use this data to build a model that, given details for any loan described above, will predict the probability that the borrower will default (also known as PD: the probability of default). Use the provided data to train a function that will estimate the probability of default for a borrower. Assuming a recovery rate of 10%, this can be used to give the expected loss on a loan.

- You should produce a function that can take in the properties of a loan and output the expected loss.
- You can explore any technique ranging from a simple regression or a decision tree to something more advanced. You can also use multiple methods and provide a comparative analysis.

## TASK 4: Bucket FICO Scores

**What you'll learn**
How to apply statistical formulas to business solutions
The importance of breaking down a large dataset using machine learning methods

**What you'll do**
Deploy detailed Python code to strategically bucket customers with various FICO scores in order to narrow in on the probability of default

### Background Information

Now that you are familiar with the portfolio and personal loans and risk are using your model as a guide to loss provisions for the upcoming year, the team now asks you to look at their mortgage book. They suspect that FICO scores will provide a good indication of how likely a customer is to default on their mortgage. Charlie wants to build a machine learning model that will predict the probability of default, but while you are discussing the methodology, she mentions that the architecture she is using requires categorical data. As FICO ratings can take integer values in a large range, they will need to be mapped into buckets. She asks if you can find the best way of doing this to allow her to analyze the data.

A FICO score is a standardized credit score created by the Fair Isaac Corporation (FICO) that quantifies the creditworthiness of a borrower to a value between 300 to 850, based on various factors. FICO scores are used in 90% of mortgage application decisions in the United States. The risk manager provides you with FICO scores for the borrowers in the bank’s portfolio and wants you to construct a technique for predicting the PD (probability of default) for the borrowers using these scores. 

### Task

#### Quantization for Rating Map: FICO Score to Credit Rating

Charlie wants to create a model that can generalize across future datasets. She aims to map **FICO scores** of borrowers to **credit ratings**, where **lower ratings** correspond to **better credit scores**. This process involves splitting the FICO score range into distinct **buckets**, with the goal of minimizing the loss in summarizing the data.

There are two main approaches to this problem: 
1. **Minimizing the mean squared error (MSE)**
2. **Maximizing the log-likelihood**

Below is a breakdown of both approaches, with relevant mathematical functions.

---

#### 1. Mean Squared Error (MSE) Minimization

This method involves minimizing the **mean squared error** between the actual FICO scores and the value assigned to each bucket. Each entry in a bucket is mapped to a representative value (often the mean or median of the bucket), and the squared error is minimized.

We aim to minimize the following equation:

\[
\text{MSE} = \sum_{i=1}^{N} \left( x_i - \hat{x}_i \right)^2
\]

Where:
- \( x_i \) is the actual FICO score of borrower \( i \).
- \( \hat{x}_i \) is the representative value for the bucket to which \( x_i \) is mapped (usually the mean of the bucket).
- \( N \) is the total number of borrowers.

The goal is to find **bucket boundaries** \( b_i \) such that the MSE is minimized for each bucket.

#### Steps:
1. Sort the FICO scores in ascending order.
2. Define a set number of buckets.
3. Assign FICO scores to buckets such that the sum of squared errors within each bucket is minimized.

---

### 2. Log-Likelihood Maximization

A more sophisticated approach involves maximizing the **log-likelihood** of the observed data given the bucket boundaries. In this case, the boundaries should not only summarize the FICO scores but also consider the **density of defaults** within each bucket.

We aim to maximize the following log-likelihood function:

\[
\text{Log-Likelihood} = \sum_{i=1}^{B} \left( k_i \cdot \log(p_i) + (n_i - k_i) \cdot \log(1 - p_i) \right)
\]

Where:
- \( B \) is the number of buckets.
- \( k_i \) is the number of **defaults** in bucket \( i \).
- \( n_i \) is the total number of borrowers in bucket \( i \).
- \( p_i = \frac{k_i}{n_i} \) is the probability of default in bucket \( i \).

This function takes into account both the **roughness** of the bucketization and the **density of defaults** within each bucket.

#### Steps:
1. Start with an initial set of boundaries for buckets.
2. Calculate the probability of default \( p_i \) for each bucket.
3. Maximize the log-likelihood by adjusting bucket boundaries such that buckets are well-balanced in terms of both size and default probability.
4. You may break the problem into **subproblems** (e.g., separate buckets for FICO ranges 0-600 and 600-850) and solve incrementally using **dynamic programming**.

---

### Dynamic Programming Approach

To solve the **log-likelihood maximization problem** efficiently, we can use **dynamic programming** by breaking the problem into subproblems. For instance:
- Divide the FICO score range (e.g., 0-850) into subranges.
- Create separate buckets for each subrange.
- Solve each subproblem incrementally by maximizing the log-likelihood within each subrange and combining the results.

#### Example:
1. Create five buckets for FICO scores between **0-600**.
2. Create five buckets for FICO scores between **600-850**.
3. Optimize each bucket's boundaries within these subranges using dynamic programming to maximize the log-likelihood.

---

### Summary

There are two main ways to approach the quantization of FICO scores into credit ratings:
1. **Minimizing the mean squared error (MSE)**: Focuses on accurately summarizing the FICO scores in each bucket by minimizing the squared error.
2. **Maximizing the log-likelihood**: Considers both the **density of defaults** and the roughness of bucketization, optimizing for a well-distributed bucketization.

Both approaches have trade-offs and are suitable depending on the goals of the analysis (e.g., whether accuracy or default prediction is more important).