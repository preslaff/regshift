# Dynamic Investment Strategies Web Application

A comprehensive web application for portfolio evaluation and regime-aware investment strategy analysis.

## Architecture

### Backend (FastAPI)
- **API Endpoints**: RESTful API for regime analysis, backtesting, and portfolio optimization
- **Real-time Processing**: Async processing for compute-intensive tasks
- **Data Integration**: FRED API, Yahoo Finance, and custom economic indicators
- **Authentication**: JWT-based user authentication and session management

### Frontend (Svelte)
- **Interactive Dashboards**: Real-time portfolio performance visualization
- **Scenario Replay**: Historical and hypothetical scenario analysis
- **Regime Visualization**: Market regime identification and transition analysis
- **Portfolio Builder**: Interactive portfolio construction and optimization tools

### Key Features

#### 🎯 Portfolio Analysis
- **Multi-Horizon Backtesting**: Test strategies across different time periods
- **Regime-Aware Optimization**: Dynamic allocation based on market regimes
- **Risk Analytics**: Comprehensive risk metrics and drawdown analysis
- **Benchmark Comparison**: Compare against static MPT and market indices

#### 📊 Interactive Dashboards
- **Performance Dashboard**: Real-time portfolio metrics and charts
- **Regime Dashboard**: Market regime identification and forecasting
- **Risk Dashboard**: Risk decomposition and stress testing
- **Scenario Analysis**: What-if analysis with custom scenarios

#### ⚙️ Configuration & Settings
- **Asset Universe**: Customizable asset selection and weights
- **Regime Methods**: Investment Clock, K-Means, Hidden Markov Models
- **Optimization Methods**: Max Sharpe, Risk Parity, Minimum Variance
- **Data Sources**: FRED API, Yahoo Finance, custom data uploads

## Getting Started

### Prerequisites
- Python 3.9+
- Node.js 16+
- FRED API Key (optional but recommended)

### Installation

#### Backend Setup
```bash
cd webapp/backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

#### Frontend Setup
```bash
cd webapp/frontend  
npm install
npm run dev
```

### Environment Variables
```env
FRED_API_KEY=your_fred_api_key
DATABASE_URL=postgresql://user:pass@localhost/db
JWT_SECRET_KEY=your_jwt_secret
```

## API Documentation

Interactive API documentation available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Development Roadmap

- [x] Core regime strategies engine
- [x] FRED API integration
- [x] Multi-horizon analysis
- [ ] FastAPI backend development
- [ ] Svelte frontend development
- [ ] User authentication system
- [ ] Real-time data streaming
- [ ] Advanced visualization components
- [ ] Deployment and scaling

## Contributing

Please read our contributing guidelines and code of conduct before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.