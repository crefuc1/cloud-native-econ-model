import numpy as np
from scipy.optimize import minimize
from typing import Dict, List
import pandas as pd

class CobbDouglasModel:
    """
    Cobb-Douglas Production Function: Y = A * K^α * L^β
    Where:
    - Y = Output
    - K = Capital
    - L = Labor
    - A = Total Factor Productivity
    - α, β = Output elasticities
    """
    
    def __init__(self, tfp: float = 1.0, alpha: float = 0.3, beta: float = 0.7):
        self.A = tfp
        self.alpha = alpha
        self.beta = beta
    
    def production(self, capital: float, labor: float) -> float:
        """Calculate output given capital and labor inputs"""
        return self.A * (capital ** self.alpha) * (labor ** self.beta)
    
    def marginal_product_capital(self, capital: float, labor: float) -> float:
        """Calculate MPK"""
        return self.alpha * self.A * (capital ** (self.alpha - 1)) * (labor ** self.beta)
    
    def marginal_product_labor(self, capital: float, labor: float) -> float:
        """Calculate MPL"""
        return self.beta * self.A * (capital ** self.alpha) * (labor ** (self.beta - 1))
    
    def optimal_allocation(self, budget: float, capital_price: float, labor_price: float) -> Dict:
        """Find optimal K and L given budget constraint"""
        
        def objective(x):
            K, L = x
            return -self.production(K, L)
        
        def budget_constraint(x):
            K, L = x
            return budget - (capital_price * K + labor_price * L)
        
        constraints = {'type': 'eq', 'fun': budget_constraint}
        bounds = [(0.01, None), (0.01, None)]
        initial_guess = [budget / (2 * capital_price), budget / (2 * labor_price)]
        
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimal_K, optimal_L = result.x
        optimal_output = self.production(optimal_K, optimal_L)
        
        return {
            'capital': round(optimal_K, 2),
            'labor': round(optimal_L, 2),
            'output': round(optimal_output, 2),
            'mpk': round(self.marginal_product_capital(optimal_K, optimal_L), 4),
            'mpl': round(self.marginal_product_labor(optimal_K, optimal_L), 4)
        }


class DemandModel:
    """Price Elasticity of Demand Model"""
    
    @staticmethod
    def demand_quantity(price: float, elasticity: float, base_price: float = 100, base_quantity: float = 1000) -> float:
        """
        Calculate quantity demanded using elasticity formula:
        Q = Q0 * (P/P0)^elasticity
        """
        return base_quantity * ((price / base_price) ** elasticity)
    
    @staticmethod
    def revenue(price: float, elasticity: float, base_price: float = 100, base_quantity: float = 1000) -> float:
        """Calculate total revenue"""
        quantity = DemandModel.demand_quantity(price, elasticity, base_price, base_quantity)
        return price * quantity
    
    @staticmethod
    def optimal_price(elasticity: float, marginal_cost: float) -> Dict:
        """
        Find optimal price using markup formula:
        P* = MC / (1 + 1/elasticity)
        """
        if elasticity >= 0:
            return {'error': 'Elasticity must be negative for downward-sloping demand'}
        
        optimal_price = marginal_cost / (1 + 1/elasticity)
        
        return {
            'optimal_price': round(optimal_price, 2),
            'markup_percentage': round(((optimal_price - marginal_cost) / marginal_cost) * 100, 2)
        }