from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from economic_model import CobbDouglasModel, DemandModel
from typing import Optional
import uvicorn

app = FastAPI(
    title="Cloud-Native Economic Model API",
    description="Production functions and demand elasticity models",
    version="1.0.0"
)

# Request Models
class ProductionRequest(BaseModel):
    capital: float = Field(gt=0, description="Capital input")
    labor: float = Field(gt=0, description="Labor input")
    tfp: Optional[float] = Field(1.0, description="Total Factor Productivity")
    alpha: Optional[float] = Field(0.3, description="Capital elasticity")
    beta: Optional[float] = Field(0.7, description="Labor elasticity")

class OptimizationRequest(BaseModel):
    budget: float = Field(gt=0, description="Total budget")
    capital_price: float = Field(gt=0, description="Price of capital")
    labor_price: float = Field(gt=0, description="Price of labor")
    tfp: Optional[float] = 1.0
    alpha: Optional[float] = 0.3
    beta: Optional[float] = 0.7

class DemandRequest(BaseModel):
    price: float = Field(gt=0, description="Price level")
    elasticity: float = Field(description="Price elasticity (should be negative)")
    base_price: Optional[float] = 100
    base_quantity: Optional[float] = 1000

class OptimalPriceRequest(BaseModel):
    elasticity: float = Field(description="Price elasticity (negative)")
    marginal_cost: float = Field(gt=0, description="Marginal cost")


# Endpoints
@app.get("/")
def read_root():
    return {
        "message": "Cloud-Native Economic Model API",
        "endpoints": [
            "/production",
            "/optimize",
            "/demand",
            "/optimal-price"
        ]
    }

@app.post("/production")
def calculate_production(request: ProductionRequest):
    """Calculate output using Cobb-Douglas function"""
    model = CobbDouglasModel(request.tfp, request.alpha, request.beta)
    output = model.production(request.capital, request.labor)
    mpk = model.marginal_product_capital(request.capital, request.labor)
    mpl = model.marginal_product_labor(request.capital, request.labor)
    
    return {
        "output": round(output, 2),
        "marginal_product_capital": round(mpk, 4),
        "marginal_product_labor": round(mpl, 4),
        "returns_to_scale": request.alpha + request.beta
    }

@app.post("/optimize")
def optimize_production(request: OptimizationRequest):
    """Find optimal capital and labor allocation"""
    model = CobbDouglasModel(request.tfp, request.alpha, request.beta)
    result = model.optimal_allocation(request.budget, request.capital_price, request.labor_price)
    return result

@app.post("/demand")
def calculate_demand(request: DemandRequest):
    """Calculate quantity demanded at given price"""
    quantity = DemandModel.demand_quantity(
        request.price, request.elasticity, 
        request.base_price, request.base_quantity
    )
    revenue = DemandModel.revenue(
        request.price, request.elasticity,
        request.base_price, request.base_quantity
    )
    
    return {
        "price": request.price,
        "quantity_demanded": round(quantity, 2),
        "total_revenue": round(revenue, 2),
        "elasticity": request.elasticity
    }

@app.post("/optimal-price")
def find_optimal_price(request: OptimalPriceRequest):
    """Calculate profit-maximizing price"""
    result = DemandModel.optimal_price(request.elasticity, request.marginal_cost)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)