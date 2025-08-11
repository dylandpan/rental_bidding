# Vancouver Rental Bidding Strategy Analyzer

## Overview

An intelligent decision-support tool that uses game theory and the expectiminimax algorithm to help tenants make optimal bidding decisions in Vancouver's competitive rental market. The system models the complex interactions between tenants, competitors, and landlords to recommend bidding strategies that maximize the chance of securing housing while minimizing overpayment.

## Features

- **Optimal Bid Calculation**: Uses expectiminimax algorithm with alpha-beta pruning to find the best initial bid
- **Market-Aware Strategies**: Adapts recommendations based on market conditions (HOT/BALANCED/COOLING)
- **Competitor Modeling**: Simulates competitor bidding behavior using Monte Carlo methods
- **Landlord Behavior Prediction**: Models landlord decision-making considering time on market and bid differences
- **Multi-Round Negotiation**: Handles best-and-final offer scenarios
- **Real Vancouver Data**: Incorporates 2024 CMHC rental market data for 10 Vancouver neighborhoods

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Required Libraries

```bash
pip install numpy
```

### Download

```bash
# Clone the repository
git clone https://github.com/dylandpan/rental_bidding.git
cd rental_bidding

# Or simply download rental_bidding.py directly
wget https://raw.githubusercontent.com/dylandpan/rental_bidding/main/rental_bidding.py
```

## Usage

### Running the Program

```bash
python rental_bidding.py
```

## Algorithm Details

### Core Components

1. **Expectiminimax Algorithm**: Multi-round game tree search with depth-3 lookahead
2. **Alpha-Beta Pruning**: Reduces computational complexity by ~50-60%
3. **Monte Carlo Simulation**: 100-1000 samples for competitor bid modeling
4. **Utility Function**: Balances securing housing reward against budget stress

### Key Parameters

- **Base utility reward**: 100 points for securing housing
- **Budget stress weight**: 2.0x penalty multiplier
- **Competitor samples**: 100 (optimization) or 1000 (simulation)
- **Search depth**: 3 rounds maximum
- **Bid strategies**: 7 levels (95%, 98%, 100%, 102%, 105%, 108%, 110% of listing)

## Data Sources

- **Primary**: CMHC Rental Market Report 2024 (Vancouver neighborhoods)
- **Validation**: Cross-referenced with RentBoard.ca, Zumper, and Vancouver.ca
- **Coverage**: 10 Vancouver neighborhoods with studio to 3-bedroom data

## Limitations

- Assumes price is the only factor (doesn't model tenant profiles, references, etc.)
- Based on average market data rather than actual winning bid data
- Simplified landlord decision model
- Command-line interface only

## Project Structure

```
rental_bidding/
│
├── rental_bidding.py          # Main program file
└── README.md                  # This file
```

## Authors

- Chen Zhuge
- Deng Pan  
- Qi Wei

---

*Note: This tool provides strategic recommendations based on market modeling and should be used as one input among many when making rental decisions. Always consider your personal circumstances and preferences.*
