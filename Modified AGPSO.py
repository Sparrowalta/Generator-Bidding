import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import copy

# --- PARAMETERS ---
NUM_FIRMS = 10
NUM_BLOCKS = 10
NUM_HOURS = 24

# Generator parameters
FIRM_PARAMS = [
    {'a2': 46.18000, 'a3': 0.00477,  'Qmax': 500},
    {'a2': 32.95070, 'a3': 0.002357, 'Qmax': 600},
    {'a2': 42.40000, 'a3': 0.004664, 'Qmax': 250},
    {'a2': 40.12000, 'a3': 0.004364, 'Qmax': 1100},
    {'a2': 41.75679, 'a3': 0.003896, 'Qmax': 585},
    {'a2': 46.26748, 'a3': 0.007919, 'Qmax': 3000},
    {'a2': 42.71000, 'a3': 0.016201, 'Qmax': 1528},
    {'a2': 44.68000, 'a3': 0.017737, 'Qmax': 2000},
    {'a2': 42.45727, 'a3': 0.006044, 'Qmax': 403},
    {'a2': 43.19774, 'a3': 0.006153, 'Qmax': 4400}
]

DEMAND_PROFILE = [
    3115, 3711, 3346, 3771, 3298, 4266, 4117, 5176, 5751, 6513, 6280, 4472,
    4510, 5142, 3424, 3287, 4501, 5236, 5790, 6084, 6561, 6411, 4411, 4664
]

BID_PRICE_MIN = 0.0
BID_PRICE_MAX = 999.0

# dBPSO Parameters
PSO_POP = 20
PSO_CYCLES_B = 15  # Cycles for price optimization
PSO_CYCLES_Q = 15  # Cycles for quantity optimization
W = 0.5
C1 = 2.0
C2 = 2.0
PENALTY_FACTOR = 1e6
STOP_AVG_PRICE_DIFF = 0.40

np.random.seed(42)

def quadratic_cost(q, a2, a3):
    """Quadratic cost function for generation"""
    return a2 * q + a3 * q * q

@dataclass
class Particle:
    """Particle class for PSO"""
    position: np.ndarray
    velocity: np.ndarray
    fitness: float
    best_position: np.ndarray
    best_fitness: float

class dBPSOOptimizer:
    def __init__(self, firm_idx: int, firm_param: Dict, competitor_bids: List, 
                 previous_market_prices: List[float] = None):
        self.firm_idx = firm_idx
        self.a2 = firm_param['a2']
        self.a3 = firm_param['a3']
        self.Qmax = firm_param['Qmax']
        self.competitor_bids = competitor_bids
        self.num_blocks = NUM_BLOCKS
        self.previous_market_prices = previous_market_prices or [50.0] * NUM_HOURS
        
        # Initialize best solutions
        self.best_prices = None
        self.best_quantities = None
        self.best_fitness = -np.inf



    ## PENALTY CALCULATIONS ##

    def penalty_function(self, prices: np.ndarray, quantities: np.ndarray) -> float:
        """Penalty function for constraint violations"""
        penalty = 0.0
        
        # Penalty for exceeding maximum capacity
        if np.sum(quantities) > self.Qmax + 1e-8:
            penalty += PENALTY_FACTOR * (np.sum(quantities) - self.Qmax)
        
        # Penalty for negative quantities
        ##if np.any(quantities < 0):
        ##    penalty += PENALTY_FACTOR * np.sum(np.abs(quantities[quantities < 0]))
        
        # Penalty for price bound violations
        if np.any(prices < BID_PRICE_MIN) or np.any(prices > BID_PRICE_MAX):
            penalty += PENALTY_FACTOR * (
                np.sum(np.abs(prices[prices < BID_PRICE_MIN] - BID_PRICE_MIN)) +
                np.sum(np.abs(prices[prices > BID_PRICE_MAX] - BID_PRICE_MAX))
            )

        ##if np.any(prices > BID_PRICE_MAX):
        ##    penalty += PENALTY_FACTOR 
        
        # Penalty for non-monotonic prices (decreasing price blocks)
        price_diffs = np.diff(prices)
        if np.any(price_diffs < 0):
            penalty += PENALTY_FACTOR * np.sum(np.abs(price_diffs[price_diffs < 0]))
        
        # Penalty for individual block exceeding total capacity
        ##if np.any(quantities > self.Qmax):
        ##    penalty += PENALTY_FACTOR * np.sum(quantities[quantities > self.Qmax] - self.Qmax)

        # NEW: Economic rationality penalty
        # Normalize quantities first to get realistic cumulative quantities
        if np.sum(quantities) > 0:
            normalized_quantities = quantities / np.sum(quantities) * self.Qmax
        else:
            normalized_quantities = quantities
        
        cumulative_q = np.cumsum(normalized_quantities)
        for i in range(len(prices)):
            marginal_cost = self.a2 + 2 * self.a3 * cumulative_q[i]
            if prices[i] < marginal_cost:
                penalty += PENALTY_FACTOR * (marginal_cost - prices[i])
        
        return penalty
    
    def calculate_profit(self, prices: np.ndarray, quantities: np.ndarray) -> float:
        """Calculate total profit for all hours with penalty"""
        total_profit = 0.0
        
        # Normalize quantities to ensure feasibility
        if np.sum(quantities) > 0:
            quantities = quantities / np.sum(quantities) * min(self.Qmax, np.sum(quantities))
        
        # Sort prices to ensure monotonicity
        sorted_indices = np.argsort(prices)
        prices_sorted = prices[sorted_indices]
        quantities_sorted = quantities[sorted_indices]
        
        for hour in range(NUM_HOURS):
            demand = DEMAND_PROFILE[hour]
            
            # Collect all market offers
            offers = []
            for fj, (competitor_prices, competitor_quantities) in enumerate(self.competitor_bids):
                if fj == self.firm_idx:
                    # Use current firm's bid
                    cum_q = np.cumsum(quantities_sorted)
                    for i in range(self.num_blocks):
                        offers.append((prices_sorted[i], cum_q[i], self.firm_idx))
                else:
                    # Use competitor's bid
                    cum_q = np.cumsum(competitor_quantities)
                    for i in range(self.num_blocks):
                        offers.append((competitor_prices[i], cum_q[i], fj))
            
            # Sort offers by price (merit order)
            offers.sort(key=lambda x: x[0])
            
            # Market clearing
            dispatch = np.zeros(NUM_FIRMS)
            dispatched = 0.0
            mcp = 0.0
            
            for price, cum_qty, firm_j in offers:
                if dispatched >= demand:
                    break
                    
                # Calculate incremental quantity for this offer
                if firm_j < len(self.competitor_bids):
                    prev_dispatch = dispatch[firm_j]
                    incremental_qty = min(cum_qty - prev_dispatch, demand - dispatched)
                    
                    if incremental_qty > 0:
                        dispatch[firm_j] += incremental_qty
                        dispatched += incremental_qty
                        mcp = price
            
            # Calculate profit for this firm in this hour
            dispatched_qty = dispatch[self.firm_idx] if self.firm_idx < len(dispatch) else 0
            revenue = mcp * dispatched_qty
            cost = quadratic_cost(dispatched_qty, self.a2, self.a3)
            total_profit += (revenue - cost)
        
        # Apply penalties
        penalty = self.penalty_function(prices, quantities)
        return total_profit - penalty

        
    def optimize_prices(self, fixed_quantities: np.ndarray) -> np.ndarray:
        """Optimize bid prices with economic lower bounds"""
        particles = []
        
        # Normalize fixed quantities first
        if np.sum(fixed_quantities) > 0:
            normalized_quantities = fixed_quantities / np.sum(fixed_quantities) * self.Qmax
        else:
            normalized_quantities = fixed_quantities
        
        # Calculate economic lower bounds using normalized quantities
        cumulative_q = np.cumsum(normalized_quantities)
        economic_lower_bounds = [self.a2 + 2 * self.a3 * q for q in cumulative_q]
        
        for _ in range(PSO_POP):
            price_variance = 5.0
            avg_prev_price = np.mean(self.previous_market_prices)
            
            base_prices = []
            for i in range(self.num_blocks):
                # Ensure price is above marginal cost
                min_price = max(
                    economic_lower_bounds[i],  # Economic constraint
                    BID_PRICE_MIN,
                    avg_prev_price - price_variance
                )
                max_price = min(BID_PRICE_MAX, avg_prev_price + price_variance)
                
                if i == 0:
                    base_prices.append(np.random.uniform(min_price, max_price))
                else:
                    # Ensure monotonicity and economic rationality
                    prev_price = base_prices[i-1]
                    min_price = max(min_price, prev_price)
                    if min_price <= max_price:
                        base_prices.append(np.random.uniform(min_price, max_price))
                    else:
                        base_prices.append(min_price)
            
            base_prices = np.array(base_prices)
            
            particle = Particle(
                position=base_prices.copy(),
                velocity=np.random.uniform(-2, 2, self.num_blocks),
                fitness=-np.inf,
                best_position=base_prices.copy(),
                best_fitness=-np.inf
            )
            particles.append(particle)
        
        # PSO optimization with consistent economic bounds
        global_best_position = particles[0].position.copy()
        global_best_fitness = -np.inf
        
        for cycle in range(PSO_CYCLES_B):
            for particle in particles:
                fitness = self.calculate_profit(particle.position, fixed_quantities)
                particle.fitness = fitness
                
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()
                
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particle.position.copy()
            
            # Update positions with consistent economic constraints
            for particle in particles:
                r1, r2 = np.random.uniform(0, 1, 2)
                particle.velocity = (W * particle.velocity + 
                                   C1 * r1 * (particle.best_position - particle.position) + 
                                   C2 * r2 * (global_best_position - particle.position))
                
                particle.position = particle.position + particle.velocity
                
                # FIXED: Apply economic bounds using normalized quantities
                for i in range(len(particle.position)):
                    marginal_cost = economic_lower_bounds[i]  # Use pre-calculated bounds
                    particle.position[i] = max(particle.position[i], marginal_cost)
                
                particle.position = np.clip(particle.position, BID_PRICE_MIN, BID_PRICE_MAX)
                particle.position = np.sort(particle.position)
                  
        return global_best_position

    
    def optimize_quantities(self, fixed_prices: np.ndarray) -> np.ndarray:
        """Optimize bid quantities given fixed prices (PQ sub-problem)"""
        # Initialize particles for quantity optimization
        particles = []
        for _ in range(PSO_POP):
            # Initialize quantities
            base_quantities = np.random.uniform(0, self.Qmax/self.num_blocks, self.num_blocks)
            base_quantities = base_quantities / np.sum(base_quantities) * self.Qmax
            
            particle = Particle(
                position=base_quantities.copy(),
                velocity=np.random.uniform(-self.Qmax/20, self.Qmax/20, self.num_blocks),
                fitness=-np.inf,
                best_position=base_quantities.copy(),
                best_fitness=-np.inf
            )
            particles.append(particle)
        
        # Find global best
        global_best_position = particles[0].position.copy()
        global_best_fitness = -np.inf
        
        # PSO iterations for quantity optimization
        for cycle in range(PSO_CYCLES_Q):
            for particle in particles:
                # Evaluate fitness
                fitness = self.calculate_profit(fixed_prices, particle.position)
                particle.fitness = fitness
                
                # Update personal best
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particle.position.copy()
            
            # Update velocities and positions
            for particle in particles:
                r1, r2 = np.random.uniform(0, 1, 2)
                particle.velocity = (W * particle.velocity + 
                                   C1 * r1 * (particle.best_position - particle.position) + 
                                   C2 * r2 * (global_best_position - particle.position))
                
                particle.position = particle.position + particle.velocity
                
                # Apply constraints
                particle.position = np.clip(particle.position, 0, self.Qmax/self.num_blocks)
                
                # Normalize to respect total capacity
                if np.sum(particle.position) > 0:
                    particle.position = particle.position / np.sum(particle.position) * self.Qmax
        
        return global_best_position
    
    def optimize(self, max_iterations: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Main dBPSO optimization loop"""
        # Initialize with random strategy
        current_quantities = np.full(self.num_blocks, self.Qmax / self.num_blocks)
        current_prices = np.linspace(45, 75, self.num_blocks)
        
        best_fitness = -np.inf
        no_improvement = 0
        
        for iteration in range(max_iterations):
            # Optimize prices given current quantities (PB sub-problem)
            optimized_prices = self.optimize_prices(current_quantities)
            
            # Optimize quantities given optimized prices (PQ sub-problem)
            optimized_quantities = self.optimize_quantities(optimized_prices)
            
            # Evaluate combined solution
            fitness = self.calculate_profit(optimized_prices, optimized_quantities)
            
            if fitness > best_fitness:
                best_fitness = fitness
                self.best_prices = optimized_prices.copy()
                self.best_quantities = optimized_quantities.copy()
                self.best_fitness = fitness
                no_improvement = 0
                
                # Update current solution
                current_prices = optimized_prices.copy()
                current_quantities = optimized_quantities.copy()
            else:
                no_improvement += 1
                if no_improvement >= 2:  # Stop if no improvement for 2 iterations
                    break
        
        return self.best_prices, self.best_quantities

class ABPSOMarketSimulation:
    """Agent-Based Market Simulation with dBPSO"""
    
    def __init__(self):
        self.firm_bids = []
        self.market_prices_per_iter = []
        
        # Initialize firm bids
        for fi, fp in enumerate(FIRM_PARAMS):
            sizes = np.full(NUM_BLOCKS, fp['Qmax'] / NUM_BLOCKS)
            prices = np.linspace(45, 70 + fi*2, NUM_BLOCKS)
            self.firm_bids.append((prices, sizes))
    
    def calculate_market_prices(self) -> List[float]:
        """Calculate market clearing prices for all hours"""
        hourly_prices = []
        
        for hour in range(NUM_HOURS):
            demand = DEMAND_PROFILE[hour]
            offers = []
            
            # Collect all offers
            for fj, (prices, quantities) in enumerate(self.firm_bids):
                cum_q = np.cumsum(quantities)
                for i in range(NUM_BLOCKS):
                    offers.append((prices[i], cum_q[i], fj))
            
            # Sort by price (merit order)
            offers.sort(key=lambda x: x[0])
            
            # Find market clearing price
            dispatched = 0
            mcp = 0
            for price, cum_qty, _ in offers:
                if dispatched + cum_qty >= demand:
                    mcp = price
                    break
                dispatched += cum_qty
            
            hourly_prices.append(mcp)
        
        return hourly_prices
    
    def run_simulation(self, max_iterations: int = 30) -> Tuple[List, List, List]:
      """Run the agent-based market simulation"""
      converged = False
      iteration = 0
      previous_market_prices = None  # Initialize as None
    
      print("Starting ABPSO Market Simulation...")
    
      while not converged and iteration < max_iterations:
          iteration += 1
          print(f"Iteration {iteration}")
        
          prev_bids = copy.deepcopy(self.firm_bids)
        
          # Each firm optimizes its strategy using dBPSO
          for fi, fp in enumerate(FIRM_PARAMS):
              print(f"  Optimizing Firm {fi+1}")
            
              # Pass previous market prices to optimizer
              optimizer = dBPSOOptimizer(fi, fp, prev_bids, previous_market_prices)
              best_prices, best_quantities = optimizer.optimize()
            
              if best_prices is not None and best_quantities is not None:
                 self.firm_bids[fi] = (best_prices, best_quantities)
        
          # Calculate market prices AFTER all firms have optimized
          current_prices = self.calculate_market_prices()
          self.market_prices_per_iter.append(current_prices)
        
          # Update previous_market_prices for next iteration
          previous_market_prices = current_prices.copy()
        
          # Check convergence
          if len(self.market_prices_per_iter) > 1:
              prev_prices = np.array(self.market_prices_per_iter[-2])
              curr_prices = np.array(current_prices)
              avg_diff = np.mean(np.abs(curr_prices - prev_prices))
            
              print(f"  Average price difference: ${avg_diff:.2f}/MWh")
              print(f"  Average market price: ${np.mean(current_prices):.2f}/MWh")
            
              if avg_diff < STOP_AVG_PRICE_DIFF:
                  converged = True
                  print(f"Converged after {iteration} iterations!")

        
      # Calculate final profits
      final_profits = self.calculate_final_profits()
        
      return self.market_prices_per_iter, final_profits, self.firm_bids
    
    def calculate_final_profits(self) -> List[float]:
        """Calculate final profits for all firms"""
        profits = []
        
        for fi, fp in enumerate(FIRM_PARAMS):
            prices, quantities = self.firm_bids[fi]
            total_profit = 0.0
            
            for hour in range(NUM_HOURS):
                demand = DEMAND_PROFILE[hour]
                offers = []
                
                # Collect all offers
                for fj, (firm_prices, firm_quantities) in enumerate(self.firm_bids):
                    cum_q = np.cumsum(firm_quantities)
                    for i in range(NUM_BLOCKS):
                        offers.append((firm_prices[i], cum_q[i], fj))
                
                # Sort and dispatch
                offers.sort(key=lambda x: x[0])
                dispatch = np.zeros(NUM_FIRMS)
                dispatched = 0.0
                mcp = 0.0
                
                for price, cum_qty, firm_j in offers:
                    if dispatched >= demand:
                        break
                    
                    prev_dispatch = dispatch[firm_j]
                    incremental_qty = min(cum_qty - prev_dispatch, demand - dispatched)
                    
                    if incremental_qty > 0:
                        dispatch[firm_j] += incremental_qty
                        dispatched += incremental_qty
                        mcp = price
                
                # Calculate profit for this firm
                dispatched_qty = dispatch[fi]
                revenue = mcp * dispatched_qty
                cost = quadratic_cost(dispatched_qty, fp['a2'], fp['a3'])
                total_profit += (revenue - cost)
            
            profits.append(total_profit)
        
        return profits

def visualize_results(market_prices_per_iter, final_profits, firm_bids):
    """Visualize simulation results"""
    
    # Plot hourly market prices
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    hours = np.arange(1, 25)
    plt.plot(hours, market_prices_per_iter[-1], 'o-', color='royalblue', linewidth=2)
    plt.xlabel('Hour')
    plt.ylabel('Market Clearing Price ($/MWh)')
    plt.title('Final Hourly Market Clearing Prices')
    plt.grid(True, alpha=0.3)
    
    # Plot firm profits
    plt.subplot(2, 2, 2)
    firms = [f'Firm {i+1}' for i in range(len(final_profits))]
    bars = plt.bar(firms, final_profits, color='darkorange', alpha=0.7)
    plt.title('Final Daily Profits by Firm')
    plt.ylabel('Profit ($/day)')
    plt.xticks(rotation=45)
    
    # Add profit values on bars
    for bar, profit in zip(bars, final_profits):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_profits)*0.01,
                f'${profit:,.0f}', ha='center', va='bottom', fontsize=8)
    
    # Plot bid curves for first 5 firms
    plt.subplot(2, 2, 3)    
    # Generate enough colors for all firms using colormap
    from matplotlib import cm
    colors = cm.get_cmap('tab10', NUM_FIRMS).colors  # 'tab10' provides 10 distinct colors
    # Plot all firms (remove the min(5, len(firm_bids)) limitation)
    for fi in range(len(firm_bids)):  # This will now iterate through all firms
        prices, quantities = firm_bids[fi]
        cum_q = np.cumsum(quantities)
        plt.step(cum_q, prices, where='post', marker='o', 
                label=f'Firm {fi+1}', color=colors[fi], linewidth=2)

    plt.xlabel('Cumulative Quantity (MWh)')
    plt.ylabel('Bid Price ($/MWh)')
    plt.title('Final Bid Curves (All 10 Firms)')
    plt.legend(ncol=2)  # Use 2 columns for legend to save space
    plt.grid(True, alpha=0.3)
        
    # Plot convergence
    if len(market_prices_per_iter) > 1:
        plt.subplot(2, 2, 4)
        avg_prices = [np.mean(prices) for prices in market_prices_per_iter]
        iterations = range(1, len(avg_prices) + 1)
        plt.plot(iterations, avg_prices, 'o-', color='green', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Average Market Price ($/MWh)')
        plt.title('Price Convergence')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# THE ECONOMIC VALIDATION FUNCTION 
def validate_economic_rationality(final_profits, firm_bids):
    """Enhanced validation with detailed marginal cost analysis"""
    print("\n" + "="*50)
    print("ECONOMIC RATIONALITY CHECK")
    print("="*50)
    
    for fi, profit in enumerate(final_profits):
        if profit < 0:
            print(f"⚠️  WARNING: Firm {fi+1} has negative profit: ${profit:,.2f}")
        else:
            print(f"✅ Firm {fi+1} profit: ${profit:,.2f}")
    
    print("\n" + "-"*50)
    print("MARGINAL COST ANALYSIS")
    print("-"*50)
    
    # Detailed analysis of bidding vs marginal cost
    for fi, (prices, quantities) in enumerate(firm_bids):
        fp = FIRM_PARAMS[fi]
        
        # Normalize quantities like in the actual calculation
        if np.sum(quantities) > 0:
            normalized_quantities = quantities / np.sum(quantities) * fp['Qmax']
        else:
            normalized_quantities = quantities
            
        cumulative_q = np.cumsum(normalized_quantities)
        
        below_cost_blocks = 0
        print(f"\nFirm {fi+1} (Qmax={fp['Qmax']} MW):")
        
        for i, (price, cum_q) in enumerate(zip(prices, cumulative_q)):
            marginal_cost = fp['a2'] + 2 * fp['a3'] * cum_q
            if price < marginal_cost:
                below_cost_blocks += 1
                print(f"  ⚠️  Block {i+1}: Price=${price:.2f} < MC=${marginal_cost:.2f} (Qty={cum_q:.1f})")
            else:
                print(f"  ✅ Block {i+1}: Price=${price:.2f} ≥ MC=${marginal_cost:.2f} (Qty={cum_q:.1f})")
        
        if below_cost_blocks == 0:
            print(f"  ✅ All blocks economically rational")

# Run the simulation
if __name__ == "__main__":
    simulation = ABPSOMarketSimulation()
    market_prices_per_iter, final_profits, firm_bids = simulation.run_simulation()
    
    # Display results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    print(f"\nFinal Market Prices ($/MWh):")
    for hour, price in enumerate(market_prices_per_iter[-1], 1):
        print(f"Hour {hour:2d}: ${price:6.2f}")
    
    print(f"\nFinal Firm Profits:")
    total_profit = 0
    for i, profit in enumerate(final_profits):
        print(f"Firm {i+1}: ${profit:10,.2f}")
        total_profit += profit
    
    print(f"\nTotal System Profit: ${total_profit:10,.2f}")
    print(f"Average Market Price: ${np.mean(market_prices_per_iter[-1]):6.2f}/MWh")

    #Calling the economic validation function
    validate_economic_rationality(final_profits, firm_bids)
    
    # Visualize results
    visualize_results(market_prices_per_iter, final_profits, firm_bids)

